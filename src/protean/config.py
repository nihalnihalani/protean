"""Validated, centralized settings layer for Protean.

This module gathers the handful of environment variables Protean reads from
scattered ``os.environ.get`` call sites into one validated, typed surface with
safe defaults. It is intentionally *additive*: existing call sites
(``grader.py``, ``fireworks_policy.py``, ``env.py``, ``bench_core.py``,
``hud_stream.py``, ``tasks.py``) are left untouched. Adoption is a follow-up
wiring step; this module merely makes a validated configuration available and
testable today.

Design notes
------------
* When ``pydantic-settings`` is installed (the optional ``config`` extra), the
  settings are backed by :class:`pydantic_settings.BaseSettings`, giving env +
  ``.env`` parsing, ``SecretStr`` masking of the API keys, and a startup-time
  ``ValidationError`` on malformed values.
* When ``pydantic-settings`` is *not* installed, we fall back to a typed
  ``dataclass`` that parses the same environment variables by hand and performs
  the same validation. The public surface (attribute names, ``load()``,
  ``fireworks_api_key_value()`` / ``hud_api_key_value()`` accessors) is
  identical across both backends, so callers never need to branch.

Environment variables centralized here
--------------------------------------
* ``FIREWORKS_API_KEY``       -> ``fireworks_api_key``
* ``HUD_API_KEY``             -> ``hud_api_key``
* ``TRITON_CACHE_DIR``        -> ``triton_cache_dir``
* ``PROTEAN_LOG``             -> ``protean_log``
* ``PROTEAN_ALLOW_DYNAMIC_MANIFEST`` -> ``protean_allow_dynamic_manifest``
* ``FIREWORKS_MODEL``         -> ``fireworks_model``  (new, with safe default)
* ``FIREWORKS_BASE_URL``      -> ``fireworks_base_url`` (new, with safe default)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

DEFAULT_FIREWORKS_MODEL = "accounts/fireworks/models/gpt-oss-120b"
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Accepted PROTEAN_LOG values: "0"/"1" sentinels or a logging level name.
_VALID_LOG_LEVELS = frozenset(
    {"", "0", "1", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL", "NOTSET"}
)


class ConfigError(ValueError):
    """Raised when a Protean environment variable holds a malformed value."""


def _validate_base_url(value: str) -> str:
    """Ensure the Fireworks base URL is a plausible http(s) endpoint."""
    if not value:
        raise ConfigError("fireworks_base_url must not be empty")
    if not value.startswith(("http://", "https://")):
        raise ConfigError(f"fireworks_base_url must start with http:// or https:// (got {value!r})")
    return value


def _validate_log(value: str) -> str:
    """Accept the sentinels and standard logging level names (case-insensitive)."""
    if value.upper() in _VALID_LOG_LEVELS or value in _VALID_LOG_LEVELS:
        return value
    raise ConfigError(f"protean_log must be empty, '0', '1', or a logging level name (got {value!r})")


def _validate_bool_flag(value: str) -> str:
    """The dynamic-manifest flag is read as ``== '1'`` by callers; keep it a clean string."""
    return value


def _env(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    return default if raw is None else raw


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised by whichever backend is installed
    from pydantic import SecretStr, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAVE_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_PYDANTIC_SETTINGS = False


@dataclass(frozen=True)
class _DataclassSettings:
    """Dataclass fallback used when ``pydantic-settings`` is unavailable.

    Parses the same environment variables and applies the same validation as the
    pydantic backend. API keys are stored as plain strings here (no ``SecretStr``
    masking), but are accessed via the same ``*_value()`` methods so callers are
    backend-agnostic.
    """

    fireworks_api_key: str | None = None
    hud_api_key: str | None = None
    triton_cache_dir: str = ""
    protean_log: str = ""
    protean_allow_dynamic_manifest: str = ""
    fireworks_model: str = DEFAULT_FIREWORKS_MODEL
    fireworks_base_url: str = DEFAULT_FIREWORKS_BASE_URL
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate only when constructed via load(); allow raw construction in tests.
        if self._validated:
            _validate_base_url(self.fireworks_base_url)
            _validate_log(self.protean_log)
            if not self.fireworks_model:
                raise ConfigError("fireworks_model must not be empty")

    def fireworks_api_key_value(self) -> str | None:
        return self.fireworks_api_key or None

    def hud_api_key_value(self) -> str | None:
        return self.hud_api_key or None


def _load_dataclass() -> _DataclassSettings:
    fw_key = os.environ.get("FIREWORKS_API_KEY")
    hud_key = os.environ.get("HUD_API_KEY")
    return _DataclassSettings(
        fireworks_api_key=fw_key if fw_key else None,
        hud_api_key=hud_key if hud_key else None,
        triton_cache_dir=_env("TRITON_CACHE_DIR"),
        protean_log=_validate_log(_env("PROTEAN_LOG")),
        protean_allow_dynamic_manifest=_validate_bool_flag(_env("PROTEAN_ALLOW_DYNAMIC_MANIFEST")),
        fireworks_model=_env("FIREWORKS_MODEL", DEFAULT_FIREWORKS_MODEL),
        fireworks_base_url=_validate_base_url(_env("FIREWORKS_BASE_URL", DEFAULT_FIREWORKS_BASE_URL)),
        _validated=True,
    )


# ``_PydanticSettings`` is the active pydantic settings *class* when the backend
# is installed, otherwise ``None``. A single ``type[Any] | None`` annotation gives
# it one consistent static type across both branches, so the ``None`` assignment
# in the fallback is well-typed (no ``type: ignore`` needed). The concrete class
# is defined under ``_PydanticSettingsCls`` and assigned here.
_PydanticSettings: type[Any] | None

if _HAVE_PYDANTIC_SETTINGS:

    class _PydanticSettingsCls(BaseSettings):
        """Pydantic-backed settings (preferred when ``pydantic-settings`` is installed)."""

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_ignore_empty=False,
            extra="ignore",
            case_sensitive=False,
        )

        fireworks_api_key: SecretStr | None = None
        hud_api_key: SecretStr | None = None
        triton_cache_dir: str = ""
        protean_log: str = ""
        protean_allow_dynamic_manifest: str = ""
        fireworks_model: str = DEFAULT_FIREWORKS_MODEL
        fireworks_base_url: str = DEFAULT_FIREWORKS_BASE_URL

        @field_validator("fireworks_base_url")
        @classmethod
        def _check_base_url(cls, v: str) -> str:
            return _validate_base_url(v)

        @field_validator("protean_log")
        @classmethod
        def _check_log(cls, v: str) -> str:
            return _validate_log(v)

        @field_validator("fireworks_model")
        @classmethod
        def _check_model(cls, v: str) -> str:
            if not v:
                raise ConfigError("fireworks_model must not be empty")
            return v

        def fireworks_api_key_value(self) -> str | None:
            """Unwrap the Fireworks key, or ``None`` if unset/empty."""
            if self.fireworks_api_key is None:
                return None
            secret = self.fireworks_api_key.get_secret_value()
            return secret or None

        def hud_api_key_value(self) -> str | None:
            """Unwrap the HUD key, or ``None`` if unset/empty."""
            if self.hud_api_key is None:
                return None
            secret = self.hud_api_key.get_secret_value()
            return secret or None

    _PydanticSettings = _PydanticSettingsCls

    def _load_pydantic() -> _PydanticSettingsCls:
        try:
            return _PydanticSettingsCls()
        except ConfigError:
            raise
        except Exception as exc:  # pydantic ValidationError -> ConfigError
            raise ConfigError(str(exc)) from exc

else:  # pragma: no cover - only when pydantic-settings is absent
    # No pydantic backend: leave the sentinel as ``None`` and route ``load()``
    # through the dataclass backend below.
    _PydanticSettings = None


# ``ProteanSettings`` names the active settings type. Both backends expose the
# same public surface (the field names + the two ``*_value()`` accessors), so a
# union lets callers stay backend-agnostic.
#
# At runtime the alias must be a usable type for ``isinstance`` / return-annotation
# purposes, so we build it only from the classes that actually exist: when the
# pydantic backend is absent ``_PydanticSettings`` is ``None`` (not a type), and
# folding it into a union would either crash or smuggle ``NoneType`` into the
# alias. Under ``TYPE_CHECKING`` we emit the full two-member union (referencing
# the concrete class) so static callers see both backends.
if TYPE_CHECKING:
    ProteanSettings = _PydanticSettingsCls | _DataclassSettings
elif _HAVE_PYDANTIC_SETTINGS:
    # ``_PydanticSettings`` is the real class on this branch, so ``|`` is safe.
    ProteanSettings = _PydanticSettings | _DataclassSettings
else:
    ProteanSettings = _DataclassSettings


def load() -> ProteanSettings:
    """Construct settings from the current environment (raises on bad values)."""
    if _HAVE_PYDANTIC_SETTINGS:
        return _load_pydantic()
    return _load_dataclass()


__all__ = [
    "ConfigError",
    "DEFAULT_FIREWORKS_BASE_URL",
    "DEFAULT_FIREWORKS_MODEL",
    "ProteanSettings",
    "load",
]
