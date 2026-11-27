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

from conftest import ROOT, FakeCtx, FakeOpener, by_email, gate, hooks, settings

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


# -- 3. API key: optional, only the per-workspace secret ---------------------------------------------
def test_key_comes_only_from_the_secret_scope_under_its_own_name(monkeypatch):
    scope = pytest.importorskip("agent.secret_scope")
    asked = []
    monkeypatch.setattr(scope, "get_secret", lambda name, default=None: asked.append(name) or " pc_ws_key ")
    monkeypatch.setenv("PRECLEARED_API_KEY", "pc_shared_env_key")
    assert settings.read_api_key() == "pc_ws_key" and asked == ["PRECLEARED_API_KEY"]
    for path in ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "os.environ" not in text and "getenv" not in text, path.name


@pytest.mark.parametrize("value", [None, "", "   "])
def test_no_key_means_the_free_allowance(monkeypatch, value):
    scope = pytest.importorskip("agent.secret_scope")
    monkeypatch.setattr(scope, "get_secret", lambda name, default=None: value)
    assert settings.read_api_key() is None


def test_an_unscoped_secret_read_falls_back_to_no_key(monkeypatch):
    scope = pytest.importorskip("agent.secret_scope")

    def unscoped(name, default=None):
        raise scope.UnscopedSecretError(name, "no scope")
    monkeypatch.setattr(scope, "get_secret", unscoped)
    assert settings.read_api_key() is None


def test_key_is_sent_only_as_a_bearer_header_and_never_shown():
    key = "pc_ws_secret"
    hook, opener = _hook(FakeCtx({"sender_country": "GB"}), lambda b, r: (401, {"message": f"bad {key}"}), key=key)
    result = hook(tool_name=SEND, args={"to": "lead@acme.co.uk"})
    assert opener.requests[0]["headers"]["authorization"] == f"Bearer {key}"
    assert key not in json.dumps(opener.requests[0]["body"]) and key not in result["message"]
    assert result["action"] == "block" and "PRECLEARED_API_KEY" in result["message"]
    hook, opener = _hook(FakeCtx({"sender_country": "GB"}), by_email({}))
    hook(tool_name=SEND, args={"to": "lead@acme.co.uk"})
    assert "authorization" not in opener.requests[0]["headers"]


def test_engine_sends_the_workspace_key(tmp_path):
    result = _engine(tmp_path)
    assert result["calls"] and all(c["auth"] == "Bearer pc_ws_key" for c in result["calls"])


# -- 4. sender_country: required, no guess; unset or unsupported holds with a clear message ------------
@pytest.mark.parametrize("value", [None, "", "DE", "FR", "Britain", 44])
def test_missing_or_unsupported_sender_country_holds_without_any_call(value):
    values = {} if value is None else {"sender_country": value}
    hook, opener = _hook(FakeCtx(values), by_email({}))
    result = hook(tool_name=SEND, args={"to": "lead@acme.co.uk"})
    assert result["action"] == "block"
    assert "isn't set, so nothing was checked" in result["message"]
    assert "plugins.entries.precleared.settings.sender_country to GB or US" in result["message"]
    assert opener.requests == []


def test_check_tool_refuses_without_sender_country():
    from conftest import tools
    opener = FakeOpener(by_email({}))
    handler = tools.make_check_handler(FakeCtx({}), opener=opener, api_key_reader=lambda: None)
    out = json.loads(handler({"email": "lead@acme.co.uk"}))
    assert "GB or US" in out["error"] and opener.requests == []


@pytest.mark.parametrize("value,expected", [("gb", "GB"), ("UK", "GB"), (" us ", "US")])
def test_sender_country_is_normalised(value, expected):
    assert settings.read_settings(FakeCtx({"sender_country": value})).sender_country == expected


def test_manifest_keeps_sender_country_required_without_a_default():
    import yaml
    spec = yaml.safe_load((ROOT / "plugin.yaml").read_text())["config_schema"]["sender_country"]
    assert spec["required"] is True and "default" not in spec and spec["choices"] == ["GB", "US"]


def test_engine_holds_every_send_until_sender_country_is_set(tmp_path):
    result = _engine(tmp_path, settings_yaml="")
    assert result["calls"] == [] and result["prompts"] == 0
    for out in (result["held"], result["clear"]):
        assert out and "sender_country to GB or US" in out


# -- 5. user-facing text never names the engine -------------------------------------------------------
def _user_strings(path):
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


# gate.py recognises the engine's own `hermes send` command line; these patterns are not shown to anyone.
MATCHERS = {"hermes", "himalaya|agentmail|swaks|hermes|curl|mail|mutt|msmtp", "(himalaya|agentmail|swaks|hermes|curl|"}


@pytest.mark.parametrize("path", sorted(ROOT.glob("*.py")), ids=lambda p: p.name)
def test_no_engine_name_in_runtime_strings(path):
    bad = [s for s in _user_strings(path) if "hermes" in s.lower() and s not in MATCHERS]
    assert not bad, bad


def test_no_engine_name_in_manifest_or_skill_text():
    import yaml
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    visible = [manifest["description"]]
    for spec in manifest["config_schema"].values():
        visible += [str(spec.get("label", "")), str(spec.get("description", ""))]
    skill = (ROOT / "skills" / "cold-email-check" / "SKILL.md").read_text().split("---", 2)
    visible += [skill[2], yaml.safe_load(skill[1])["description"]]
    assert not [v for v in visible if "hermes" in v.lower()]
    assert "the assistant" in manifest["description"]


def test_engine_send_command_is_reported_without_the_engine_name():
    hook, _ = _hook(FakeCtx({"sender_country": "GB"}), by_email({"lead@acme.co.uk": "blocked"}))
    result = hook(tool_name="terminal", args={"command": 'hermes send --to email:lead@acme.co.uk "hi"'})
    assert result["action"] == "block" and "via send command" in result["message"]
    assert "hermes" not in result["message"].lower()


def test_user_agent_is_neutral():
    from conftest import client
    assert "hermes" not in client.USER_AGENT.lower()


def test_skill_does_not_push_a_key_signup_or_a_purchase():
    text = (ROOT / "skills" / "cold-email-check" / "SKILL.md").read_text()
    assert "get-key" not in text and "pricing" not in text and "Paid" not in text


# -- 6. shipped surface: no telemetry, self-update, report-a-problem or CI ----------------------------
def test_only_the_cold_email_skill_ships():
    assert [p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md")] == ["cold-email-check"]


def test_runtime_talks_only_to_precleared_and_runs_nothing():
    for path in ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for bad in ("subprocess", "os.system", "pip install", "telemetry", "posthog", "sentry", "self_update"):
            assert bad not in text, (path.name, bad)
    from conftest import client
    assert client.BASE_URL == "https://precleared.dev"
    assert not (ROOT / ".github").exists() and not (ROOT / "tests" / "test_live.py").exists()


def test_engine_registers_the_one_skill(tmp_path):
    assert _engine(tmp_path)["skills"] == ["precleared:cold-email-check"]


# -- 6. per-address override: one known contact, never their whole (freemail) domain --------------
def test_ignore_addresses_skips_only_that_exact_address():
    ctx = FakeCtx({"sender_country": "US", "ignore_addresses": ["Pat@Gmail.com", "not-an-address", 7]})
    assert settings.read_settings(ctx).ignore_addresses == ["pat@gmail.com"]
    hook, opener = _hook(ctx, by_email({"sam@gmail.com": "consent_needed"}))
    assert hook(tool_name=SEND, args={"to": "pat@gmail.com"}) is None
    assert opener.requests == []
    result = hook(tool_name=SEND, args={"to": ["pat@gmail.com", "sam@gmail.com"]})
    assert result["action"] == "block"
    assert [r["body"]["recipient_email"] for r in opener.requests] == ["sam@gmail.com"]


def test_block_text_suggests_the_address_never_a_domain():
    assert "ignore_addresses" in hooks.BLOCK_TAIL
    assert "ignore_domains" not in hooks.BLOCK_TAIL
    assert "whole domain" in hooks.BLOCK_TAIL


def test_manifest_points_at_the_fork_and_describes_ask_honestly():
    import yaml

    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert manifest["homepage"] == "https://github.com/enckequity/hermes-precleared"
    assert "enckequity" in manifest["author"]
    assert manifest["config_schema"]["ignore_addresses"]["type"] == "list"
    mode = manifest["config_schema"]["mode"]
    # The engine's settings editor rejects a value outside choices, but an "ask" already in config.yaml (an older
    # version, a hand edit) never passes through it, so the plugin reads it as block.
    assert "ask" not in mode["choices"] and settings.read_settings(FakeCtx({"mode": "ask"})).mode == "block"
    assert settings.SENDER_COUNTRY_PATH == "plugins.entries.precleared.settings.sender_country"
    assert not hasattr(settings, "SET_COMMAND")


def test_ignore_addresses_never_widens_to_a_domain():
    ctx = FakeCtx({"sender_country": "US", "ignore_addresses": ["@gmail.com", "gmail.com", " *@gmail.com "]})
    assert settings.read_settings(ctx).ignore_addresses == ["@gmail.com", "*@gmail.com"]
    hook, opener = _hook(ctx, by_email({"sam@gmail.com": "consent_needed"}))
    assert hook(tool_name=SEND, args={"to": "sam@gmail.com"})["action"] == "block"
    assert [r["body"]["recipient_email"] for r in opener.requests] == ["sam@gmail.com"]
    assert gate.is_skipped_address(" Pat@GMAIL.com ", (), ["pat@gmail.com"])
