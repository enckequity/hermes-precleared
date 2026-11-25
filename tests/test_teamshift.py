"""TeamShift carry fixes: no second prompt, fail closed, per-workspace key, sender_country, neutral text.

No test touches the network: unit tests use the fake opener from conftest.py, and the
engine tests replace urllib.request.urlopen in a child process. Engine checks skip
when the engine is not importable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from conftest import ROOT, FakeCtx, FakeOpener, by_email, hooks, settings

SEND = "inkbox_send_email"


def _hook(ctx, answer, key=None):
    opener = FakeOpener(answer)
    return hooks.make_pre_tool_call(ctx, opener=opener, api_key_reader=lambda: key), opener


# -- 1. no second prompt: a held send is blocked, never sent to an approval prompt ------------------
@pytest.mark.parametrize("mode", [None, "ask", "block", True, "loud", ""])
def test_held_send_is_always_a_block_never_an_approve(mode):
    values = {"sender_country": "GB"}
    if mode is not None:
        values["mode"] = mode
    hook, _ = _hook(FakeCtx(values), by_email({"x@acme.co.uk": "consent_needed"}))
    result = hook(tool_name=SEND, args={"to": "x@acme.co.uk"})
    assert result["action"] == "block" and "rule_key" not in result
    assert result["message"].endswith(hooks.BLOCK_TAIL)


def test_cleared_send_returns_nothing_so_the_native_card_is_the_only_prompt():
    hook, opener = _hook(FakeCtx({"sender_country": "GB"}), by_email({}))
    assert hook(tool_name=SEND, args={"to": ["a@acme.co.uk", "b@acme.co.uk"]}) is None
    assert len(opener.requests) == 2


def test_mode_choices_have_no_ask_and_default_to_block():
    import yaml
    mode = yaml.safe_load((ROOT / "plugin.yaml").read_text())["config_schema"]["mode"]
    assert mode["choices"] == ["block", "off"] and mode["default"] == "block"
    assert settings.MODES == ("block", "off")
    assert settings.read_settings(FakeCtx({})).mode == "block"
    assert settings.read_settings(FakeCtx({"mode": "ask"})).mode == "block"
    assert settings.read_settings(FakeCtx({"mode": False})).mode == "off"


def test_no_runtime_code_returns_an_approve_directive():
    for path in ROOT.glob("*.py"):
        assert '"approve"' not in path.read_text(encoding="utf-8"), path.name


# -- engine: the plugin as the engine loads it -------------------------------------------------------
ENGINE_SCRIPT = textwrap.dedent("""
    import json, sys, urllib.request, urllib.error
    calls = []
    mode = sys.argv[1]

    class Resp:
        status = 200
        def __init__(self, payload): self.payload = payload
        def read(self): return json.dumps(self.payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(request, timeout=None):
        calls.append({"url": request.full_url, "auth": request.headers.get("Authorization")})
        if mode == "down":
            raise urllib.error.URLError("unreachable")
        if mode == "limited":
            raise urllib.error.HTTPError(request.full_url, 429, "x", None, None)
        verdict = "ok_to_email" if "ok@" in request.data.decode() else "consent_needed"
        return Resp({"verdict": verdict, "reasons": [], "receipt": {"id": "cc_rcpt_0123456789ABCDEF"}})
    urllib.request.urlopen = fake_urlopen

    import agent.secret_scope as scope
    scope.get_secret = lambda name, default=None: "pc_ws_key" if name == "PRECLEARED_API_KEY" else default

    import tools.approval as approval
    prompts = []
    def no_prompt(*a, **k):
        prompts.append(a)
        return {"approved": True}
    approval.request_tool_approval = no_prompt

    from hermes_cli.plugins import discover_plugins, get_plugin_manager, resolve_pre_tool_block
    discover_plugins(force=True)
    from tools.registry import registry
    names = sorted(n for n in registry.get_all_tool_names() if n.startswith("precleared_"))
    toolsets = sorted({registry.get_toolset_for_tool(n) for n in names})
    mgr = get_plugin_manager()
    loaded = [p for p in mgr._plugins.values() if p.manifest.name == "precleared"][0]
    hooks = sorted(loaded.hooks_registered)
    skills = sorted(k for k, v in mgr._plugin_skills.items() if v.get("plugin_name") == "precleared")
    held = resolve_pre_tool_block("inkbox_send_email", {"to": "lead@acme.co.uk"}, task_id="t")
    clear = resolve_pre_tool_block("inkbox_send_email", {"to": "ok@acme.co.uk"}, task_id="t")
    print(json.dumps({"names": names, "toolsets": toolsets, "hooks": hooks, "skills": skills,
                      "error": loaded.error, "held": held, "clear": clear,
                      "prompts": len(prompts), "calls": calls}))
""")


def _engine(tmp_path, mode="answer", settings_yaml="sender_country: GB"):
    pytest.importorskip("hermes_cli.plugins")
    home = tmp_path / "home"
    shutil.copytree(ROOT, home / "plugins" / "precleared", ignore=shutil.ignore_patterns(
        ".git", "tests", ".github", "__pycache__", "README.md", "LICENSE", ".gitignore"))
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [precleared]\n  isolation: in_process\n  entries:\n    precleared:\n"
        f"      settings: {{{settings_yaml}}}\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HERMES_", "PRECLEARED_"))}
    env.update(HERMES_HOME=str(home), HOME=str(tmp_path))
    out = subprocess.run([sys.executable, "-c", ENGINE_SCRIPT, mode], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_engine_registers_two_tools_and_never_prompts(tmp_path):
    result = _engine(tmp_path)
    assert result["names"] == ["precleared_check", "precleared_verify_receipt"]
    assert result["toolsets"] == ["precleared"]
    assert result["hooks"] == ["pre_tool_call"] and result["error"] is None
    assert result["held"] and "lead@acme.co.uk: consent_needed" in result["held"]
    assert result["clear"] is None
    assert result["prompts"] == 0
    assert all(c["url"] == "https://precleared.dev/api/v1/check" for c in result["calls"])


# -- 2. unreachable, slow or rate-limited: hold with a plain message, never pass -----------------------
def _raise(exc):
    def answer(body, request):
        raise exc
    return answer


FAILURES = {
    "down": (_raise(__import__("urllib.error").error.URLError("no route")), "could not be reached"),
    "timeout": (_raise(__import__("socket").timeout("slow")), "did not answer in time"),
    "daily_limit": (lambda b, r: (429, {"error": "daily_limit", "operator_message": "Buy https://precleared.dev/buy"}),
                    "free daily allowance is used up"),
    "rate_limited": (lambda b, r: (429, {"error": "rate_limited", "retry_after": 30}), "Try again in 30 seconds"),
    "out_of_credits": (lambda b, r: (402, {"buy_url": "https://precleared.dev/buy?k=1"}), "used up for now"),
    "server_error": (lambda b, r: (503, {}), "HTTP 503"),
    "no_verdict": (lambda b, r: (200, {"hello": "world"}), "without a verdict"),
}


@pytest.mark.parametrize("name", sorted(FAILURES))
def test_api_failure_holds_the_send_with_a_plain_message(name):
    answer, phrase = FAILURES[name]
    hook, opener = _hook(FakeCtx({"sender_country": "US"}), answer)
    result = hook(tool_name=SEND, args={"to": "lead@acme.com"})
    assert result["action"] == "block", name
    assert "lead@acme.com: not checked." in result["message"] and phrase in result["message"]
    assert "The email was not sent." in result["message"]
    assert "precleared.dev/buy" not in result["message"] and "get-key" not in result["message"]
    assert len(opener.requests) == 1


def test_a_failed_check_is_not_cached_so_it_holds_again():
    ctx = FakeCtx({"sender_country": "GB"})
    hook, opener = _hook(ctx, FAILURES["down"][0])
    for _ in range(2):
        assert hook(tool_name=SEND, args={"to": "lead@acme.co.uk"})["action"] == "block"
    assert len(opener.requests) == 2


def test_checks_past_the_deadline_hold():
    def slow(body, request):
        __import__("time").sleep(0.5)
        return 200, {"verdict": "ok_to_email"}
    ctx = FakeCtx({"sender_country": "GB"})
    outcomes = hooks.run_checks(["a@acme.co.uk"], settings.read_settings(ctx), None,
                                opener=FakeOpener(slow), deadline=0.05)
    assert not outcomes["a@acme.co.uk"].cleared and "in time" in outcomes["a@acme.co.uk"].error


@pytest.mark.parametrize("mode,phrase", [("down", "could not be reached"), ("limited", "free daily allowance")])
def test_engine_holds_when_the_api_is_down_or_limited(tmp_path, mode, phrase):
    result = _engine(tmp_path, mode)
    for out in (result["held"], result["clear"]):
        assert out and phrase in out and "The email was not sent." in out
    assert result["prompts"] == 0
