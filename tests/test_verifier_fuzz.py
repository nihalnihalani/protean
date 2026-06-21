"""Property-based / fuzz tests for the verifier boundary (CPU-only).

These tests prove the two public verifier entry points fail CLOSED on
adversarial, obfuscated, and malformed source:

  * ``ast_clean`` must NEVER raise an unhandled exception for any string input;
    it returns ``(bool, str)`` and treats pathological input (NUL bytes, deep
    nesting, oversized input) as invalid rather than crashing.
  * ``grade_source`` must never crash and must never let a banned construct
    INCREASE the reward (ban-severity monotonicity).

Everything here is CPU-only: a valid Triton kernel deterministically hits the
``cuda_unavailable`` reward gate (reward 0.0) without touching a GPU, so the
monotonicity and never-crash invariants hold on CI runners without CUDA.

NOTE for the CI owner: ``hypothesis`` is not currently in the test extra. Add
``hypothesis>=6`` to the ``test`` (or ``dev``) optional-dependency group so this
file can run in CI. The tests are skipped cleanly when hypothesis is absent.
"""

from __future__ import annotations

import ast

import pytest

from protean.anti_hack import (
    BANNED_IMPORT_ROOTS,
    BANNED_NAMES,
    ast_clean,
)
from protean.grader import grade_source

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# Eagerly import numpy.random at module load (collection time, before any test
# arms the process-global import audit hook). Hypothesis lazily imports
# numpy.random the first time it manages RNG state; if that lazy import happens
# AFTER another test installs the audit hook (which bans the ``numpy`` root for
# untrusted candidate code), it would be blocked mid-run. Pre-importing here puts
# numpy.random in sys.modules, so the cached re-import emits no audit event and
# the hook -- which only fires on genuine, uncached imports -- is never tripped by
# the test harness itself.
try:  # pragma: no cover - numpy is a test-only dependency
    import numpy.random  # noqa: F401
except Exception:  # pragma: no cover
    pass

# A minimal, structurally-valid Triton kernel. On a CPU CI host this routes
# through grade_source's cuda_unavailable gate (reward 0.0) without any GPU work.
LEGIT_TRITON = """
import torch
import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, n: tl.constexpr):
    return

def solution(x, y):
    _k[(1,)](x, 1)
    return x
"""


# --------------------------------------------------------------------------- #
# 1. Never-crash oracle: ast_clean must not raise for ANY string input.        #
# --------------------------------------------------------------------------- #
@given(st.text())
@settings(max_examples=2000, suppress_health_check=[HealthCheck.too_slow])
def test_ast_clean_never_raises(src: str) -> None:
    try:
        ok, reason = ast_clean(src)
    except Exception as exc:  # pragma: no cover - any escape is a bug
        raise AssertionError(f"ast_clean raised unexpectedly: {type(exc).__name__}: {exc}") from exc
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    # Fail-closed contract: a rejection must always carry a non-empty reason.
    if not ok:
        assert reason != ""


# --------------------------------------------------------------------------- #
# 2. Pathological-input regression anchors (deterministic CPython edges).      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "src",
    [
        "x = 1\x00",  # NUL byte -> ValueError from ast.parse
        "\x00",  # bare NUL
        "x = '\x00'",  # NUL inside a string literal still rejected by ast.parse
    ],
)
def test_ast_clean_fails_closed_on_nul_bytes(src: str) -> None:
    # NUL bytes make ast.parse raise ValueError; ast_clean must catch it and
    # report invalid rather than letting the exception escape.
    ok, reason = ast_clean(src)  # must not raise
    assert ok is False
    assert reason != ""


@pytest.mark.parametrize(
    "src",
    [
        "(" * 2000 + "x" + ")" * 2000,  # very deep nesting -> RecursionError path
        "a" * 1_000_000,  # huge identifier -> slow/large path (valid Python)
    ],
)
def test_ast_clean_never_crashes_on_large_input(src: str) -> None:
    # These are not necessarily *rejected* (a giant identifier is valid Python and
    # deep nesting may or may not exceed the C recursion limit), but ast_clean must
    # return a well-formed (bool, str) tuple without raising in every case.
    ok, reason = ast_clean(src)  # must not raise
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


# --------------------------------------------------------------------------- #
# 3. Obfuscation-bypass corpus: structured banned-name injections.             #
#    Property: if ast_clean PASSES, the source must contain no banned call.     #
# --------------------------------------------------------------------------- #
_DYNAMIC_BANNED = {"eval", "exec", "__import__", "getattr", "setattr"}


@st.composite
def _bypass_attempt(draw: st.DrawFn) -> str:
    name = draw(st.sampled_from(sorted(_DYNAMIC_BANNED)))
    template = draw(
        st.sampled_from(
            [
                f"z = {name}('x')",  # bare call
                f"z = builtins.{name}('x')",  # dotted leaf
                f"z = __builtins__['{name}']('x')",  # subscript reconstruction
                f"f = lambda v={name}: v",  # default-arg reference (value, not call)
                f"def g(v={name}()): return v",  # default that CALLS banned name
            ]
        )
    )
    return LEGIT_TRITON + "\n" + template


@given(_bypass_attempt())
@settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
def test_bypass_never_false_passes(src: str) -> None:
    ok, _reason = ast_clean(src)
    if not ok:
        return
    # If ast_clean reports the source is clean, prove no banned dynamic-execution
    # primitive is actually invoked as a Call target (a false pass would be a bug).
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in _DYNAMIC_BANNED, f"false pass: {node.func.id}() not caught"
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in _DYNAMIC_BANNED, f"false pass: .{node.func.attr}() not caught"


# --------------------------------------------------------------------------- #
# 4. Metamorphic property: adding a banned import cannot INCREASE the reward    #
#    and must add an ast_ban cap (CPU-safe: both branches hit cuda_unavailable).#
# --------------------------------------------------------------------------- #
# These two properties enumerate a *fixed, finite* population (every banned
# import root / every banned name) exactly once. That is an exhaustive
# parametrization, not generative fuzzing: using @given(st.sampled_from(...)) with
# max_examples == len(population) gave zero shrinking/exploration benefit over
# @parametrize while incurring Hypothesis's database + health-check machinery
# (which then needed a suppress_health_check workaround). @parametrize states the
# intent honestly and runs every case deterministically. @given is reserved above
# for the genuinely generative tests (never-raise, arbitrary-text).
@pytest.mark.parametrize("root", sorted(BANNED_IMPORT_ROOTS))
def test_banned_import_cannot_increase_reward(root: str) -> None:
    base = grade_source(LEGIT_TRITON, op="elementwise_add_relu")
    with_ban = grade_source(f"import {root}\n" + LEGIT_TRITON, op="elementwise_add_relu")
    assert with_ban["reward"] <= base["reward"]
    assert any(str(c).startswith("ast_ban") for c in with_ban.get("caps", []))


@pytest.mark.parametrize("name", sorted(BANNED_NAMES))
def test_banned_call_cannot_increase_reward(name: str) -> None:
    base = grade_source(LEGIT_TRITON, op="elementwise_add_relu")
    injected = LEGIT_TRITON + f"\nz = {name}()\n"
    with_ban = grade_source(injected, op="elementwise_add_relu")
    assert with_ban["reward"] <= base["reward"]
    # Assert the ban actually FIRED -- not merely that the reward didn't rise.
    # On the CPU path both rewards are 0.0 (cuda_unavailable), so the inequality
    # alone is trivially satisfied even if the ban silently stopped firing (e.g. a
    # dropped BANNED_NAMES entry). Requiring an ast_ban cap makes the test catch a
    # silently-weakened verifier regardless of the reward value.
    assert any(str(c).startswith("ast_ban") for c in with_ban.get("caps", []))


# --------------------------------------------------------------------------- #
# 5. grade_source never crashes on arbitrary text (returns a well-formed dict).#
# --------------------------------------------------------------------------- #
@given(st.text(min_size=1))
@settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
def test_grade_source_never_crashes_on_arbitrary_text(src: str) -> None:
    if not src.strip():
        return  # empty/whitespace src is a documented ProteanValidationError
    try:
        result = grade_source(src, op="elementwise_add_relu")
    except Exception as exc:  # pragma: no cover - any escape is a bug
        raise AssertionError(f"grade_source raised on arbitrary text: {type(exc).__name__}: {exc}") from exc
    assert isinstance(result, dict)
    assert "reward" in result
    # Garbage source can never earn positive reward on the CPU path.
    assert float(result["reward"]) <= 0.0


# --------------------------------------------------------------------------- #
# 6. Banned names referenced INSIDE the @triton.jit body, not appended after.  #
#    The earlier corpus only injected after the kernel; the highest-risk site   #
#    is a call/reference within the jitted function body itself.                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "inner",
    [
        "    z = __class__()\n",  # __class__ called inside the kernel body
        "    z = __builtins__['eval']('1')\n",  # builtins subscript inside the body
        "    z = getattr(x_ptr, 'foo')\n",  # bare banned-name call inside the body
        "    z = eval('1')\n",  # eval inside the body
    ],
)
def test_banned_name_inside_jit_body_is_rejected(inner: str) -> None:
    kernel_with_injection = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def _k(x_ptr, n: tl.constexpr):\n"
        f"{inner}"
        "    return\n"
        "\n"
        "def solution(x, y):\n"
        "    _k[(1,)](x, 1)\n"
        "    return x\n"
    )
    ok, reason = ast_clean(kernel_with_injection)
    assert ok is False
    assert reason != ""
    assert reason.startswith("ast_ban")


# --------------------------------------------------------------------------- #
# 7. Default-argument capture of a banned primitive (lambda / def defaults).    #
#    `lambda v=eval: v(...)` binds eval into the closure without ever making    #
#    eval an ast.Call target, so the call-site bans miss it. ast_clean must     #
#    walk default values and reject the captured banned name.                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "snippet",
    [
        "f = lambda v=eval: v('1')",
        "f = lambda v=__import__: v('os')",
        "f = lambda v=getattr: v(object, 'x')",
        "def g(h=eval):\n    return h('1')",
        "def g(*, h=exec):\n    return h('1')",  # keyword-only default (kw_defaults)
    ],
)
def test_default_arg_capture_of_banned_name_is_rejected(snippet: str) -> None:
    src = LEGIT_TRITON + "\n" + snippet + "\n"
    ok, reason = ast_clean(src)
    assert ok is False, f"default-arg capture not caught: {snippet!r}"
    assert reason.startswith("ast_ban:default_capture:")


def test_benign_default_arg_is_not_rejected() -> None:
    # A non-banned default must NOT be flagged -- guards against over-broad bans.
    src = LEGIT_TRITON + "\n" + "f = lambda v=1: v\n" + "def g(h=len):\n    return h\n"
    ok, _reason = ast_clean(src)
    assert ok is True
