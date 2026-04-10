"""Regression tests for the silent litellm deadlock fix.

These tests pin down the failure mode that motivated PR #384:

  1. ``litellm.acompletion`` hangs forever waiting on a half-closed httpx
     socket.
  2. ``asyncio.wait_for`` cancels the parent coroutine but the underlying
     httpx pool is left in a bad state.
  3. Every subsequent ``acompletion`` call grabs the same stale connection
     and hangs forever — silent deadlock; py-spy shows all asyncio worker
     threads idle and zero active Python frames anywhere.

The fix has three load-bearing pieces, each tested below:

  * ``litellm_params['timeout']`` is set so litellm/httpx aborts the socket
    itself when it does honor the parameter.
  * An ``asyncio.wait_for`` safety net at 2× the configured timeout fires
    even if litellm ignores its own timeout (it currently does).
  * On timeout, ``_reset_litellm_http_clients`` clears every known
    module-level cache so the next call opens a fresh pool, breaking the
    deadlock cycle.

If any of these regress, an extract_all_entities-style workload will start
hanging silently in production again. Please don't delete these tests
without re-running an end-to-end deep-research session against the real
litellm pool.
"""

from __future__ import annotations

import asyncio
import copy
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentfield.agent_ai import AgentAI, _reset_litellm_http_clients
from tests.helpers import StubAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _DeadlockTestConfig:
    """Minimal AI config that exercises the litellm + rate-limiter path."""

    def __init__(self):
        self.model = "openai/gpt-4"
        self.temperature = 0.1
        self.max_tokens = 100
        self.top_p = 1.0
        self.stream = False
        self.response_format = "auto"
        self.fallback_models = []
        self.final_fallback_model = None
        self.enable_rate_limit_retry = False  # bypass retries; we want to see raw behavior
        self.rate_limit_max_retries = 0
        self.rate_limit_base_delay = 0.0
        self.rate_limit_max_delay = 0.0
        self.rate_limit_jitter_factor = 0.0
        self.rate_limit_circuit_breaker_threshold = 1
        self.rate_limit_circuit_breaker_timeout = 1
        self.auto_inject_memory = []
        self.model_limits_cache = {}
        self.audio_model = "tts-1"
        self.vision_model = "dall-e-3"

    def copy(self, deep=False):
        return copy.deepcopy(self)

    async def get_model_limits(self, model=None):
        return {"context_length": 1000, "max_output_tokens": 100}

    def get_litellm_params(self, **overrides):
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        params.update(overrides)
        return params


@pytest.fixture
def fast_timeout_agent():
    """Stub agent whose `llm_call_timeout` is short enough to test in real time."""
    agent = StubAgent()
    agent.ai_config = _DeadlockTestConfig()
    agent.memory = SimpleNamespace()
    # 0.2s timeout → asyncio safety net at 0.4s. Tests run in well under a second.
    agent.async_config = SimpleNamespace(
        llm_call_timeout=0.2,
        connection_pool_size=4,
        connection_pool_per_host=4,
    )
    return agent


def _install_litellm_stub(monkeypatch, acompletion_side_effect):
    """Install a fake `litellm` module with a controllable `acompletion`."""
    module = types.ModuleType("litellm")
    module.acompletion = acompletion_side_effect

    # Cached client attributes that `_reset_litellm_http_clients` should wipe.
    # Pre-populate them so we can assert they're cleared post-timeout.
    module.module_level_client = MagicMock(name="module_level_client")
    module.module_level_aclient = MagicMock(name="module_level_aclient")
    module.aclient_session = MagicMock(name="aclient_session")
    module.client_session = MagicMock(name="client_session")
    module.in_memory_llm_clients_cache = MagicMock(name="in_memory_llm_clients_cache")
    module.in_memory_llm_clients_cache.clear = MagicMock(name="cache_clear")

    utils_module = types.ModuleType("utils")
    utils_module.get_max_tokens = lambda model: 8192
    utils_module.token_counter = lambda model, messages: 10
    utils_module.trim_messages = lambda messages, model, max_tokens: messages
    module.utils = utils_module

    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setitem(sys.modules, "litellm.utils", utils_module)
    monkeypatch.setattr("agentfield.agent_ai.litellm", module, raising=False)
    return module


def _make_chat_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, audio=None))]
    )


# ---------------------------------------------------------------------------
# 1. _reset_litellm_http_clients unit behavior
# ---------------------------------------------------------------------------


def test_reset_litellm_http_clients_clears_known_caches(monkeypatch):
    """The reset helper must clear every module-level client attribute we
    know litellm uses to pool connections. If litellm renames or removes one,
    this test will catch it before production does."""

    fake_litellm = types.ModuleType("litellm")
    cleared = {"cache_called": False}

    class _ClearableCache:
        def clear(self):
            cleared["cache_called"] = True

    fake_litellm.module_level_client = object()
    fake_litellm.module_level_aclient = object()
    fake_litellm.aclient_session = object()
    fake_litellm.client_session = object()
    fake_litellm.in_memory_llm_clients_cache = _ClearableCache()

    _reset_litellm_http_clients(fake_litellm)

    assert fake_litellm.module_level_client is None
    assert fake_litellm.module_level_aclient is None
    assert fake_litellm.aclient_session is None
    assert fake_litellm.client_session is None
    assert cleared["cache_called"] is True, (
        "in_memory_llm_clients_cache.clear() must be called so the next "
        "litellm.acompletion gets a fresh client pool."
    )


def test_reset_litellm_http_clients_tolerates_missing_attrs():
    """Litellm versions vary; the reset must not raise on missing attributes."""
    fake_litellm = types.ModuleType("litellm")
    # Empty module — none of the cache attrs exist.
    _reset_litellm_http_clients(fake_litellm)  # must not raise


def test_reset_litellm_http_clients_tolerates_none_module():
    """Defensive: passing None must be a no-op, not a crash."""
    _reset_litellm_http_clients(None)  # must not raise


# ---------------------------------------------------------------------------
# 2. End-to-end deadlock-recovery via _make_litellm_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hanging_acompletion_triggers_timeout_and_pool_reset(
    monkeypatch, fast_timeout_agent
):
    """The smoking gun. Reproduces the production deadlock in miniature:

      1. acompletion hangs (asyncio.Event that never sets, like a half-closed
         httpx socket).
      2. The asyncio.wait_for safety net at 2 × llm_call_timeout fires.
      3. _reset_litellm_http_clients is invoked (cached clients become None).
      4. A subsequent acompletion call returns successfully (the fix worked).
    """
    call_count = {"n": 0}
    never_set = asyncio.Event()  # never set → simulates a hung HTTP read

    async def acompletion_side_effect(**params):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call hangs forever, like a half-closed socket.
            await never_set.wait()
            return _make_chat_response("never reached")
        # Second call (after the pool reset) succeeds.
        return _make_chat_response("recovered")

    stub_module = _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    # 1) First call must time out via the safety net (2× llm_call_timeout = 0.4s),
    #    not hang forever.
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(ai.ai("hello"), timeout=2.0)

    # 2) Pool reset must have fired.
    assert stub_module.module_level_aclient is None, (
        "litellm.module_level_aclient should be cleared after a timeout — "
        "without this, the next call grabs the stuck client and deadlocks."
    )
    assert stub_module.module_level_client is None
    assert stub_module.aclient_session is None
    assert stub_module.client_session is None

    # 3) The next call must succeed (i.e., we are not in a permanent
    #    deadlocked state). This is the actual production-relevant assertion:
    #    one slow request must not poison every subsequent request.
    never_set.set()  # let any lingering coroutine unblock for clean shutdown
    result = await asyncio.wait_for(ai.ai("hello again"), timeout=2.0)
    assert hasattr(result, "text")
    assert result.text == "recovered"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_litellm_params_includes_request_timeout(monkeypatch, fast_timeout_agent):
    """litellm should always be called with an explicit `timeout` parameter
    matching `async_config.llm_call_timeout`. If litellm gains proper timeout
    support in a future version, this is what makes us pick it up — and even
    today, it's the only thing that lets httpx abort the socket cleanly."""
    captured: Dict[str, Any] = {}

    async def acompletion_side_effect(**params):
        captured.update(params)
        return _make_chat_response("ok")

    _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    await ai.ai("hello")

    assert "timeout" in captured, (
        "litellm.acompletion must be called with a `timeout` parameter so "
        "httpx can abort the socket itself, not just our asyncio coroutine."
    )
    assert captured["timeout"] == fast_timeout_agent.async_config.llm_call_timeout


@pytest.mark.asyncio
async def test_safety_net_fires_within_two_times_llm_call_timeout(
    monkeypatch, fast_timeout_agent
):
    """Bound the worst-case wall-clock time. If the safety-net multiplier
    regresses (e.g., someone bumps it to 10× or removes it), production hangs
    that *do* happen will be invisible for many minutes — exactly the bug we
    were chasing.

    With llm_call_timeout=0.2s the cancel must land well under 1.0s.
    """
    never_set = asyncio.Event()

    async def acompletion_side_effect(**params):
        await never_set.wait()
        return _make_chat_response("never")

    _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    loop = asyncio.get_event_loop()
    started = loop.time()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        # Outer wait_for is generous; the SDK's own safety net should fire first.
        await asyncio.wait_for(ai.ai("hello"), timeout=2.0)
    elapsed = loop.time() - started

    # llm_call_timeout=0.2 → safety net 0.4s. Allow generous slack for CI.
    assert elapsed < 1.5, (
        f"Safety net took {elapsed:.2f}s — expected < 1.5s. Something has "
        f"regressed the asyncio.wait_for(timeout=2× llm_call_timeout) logic."
    )

    never_set.set()
