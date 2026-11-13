from __future__ import annotations

import os
import ipaddress
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urlsplit

MODEL_NAME = "gpt-3.5-turbo-1106"
PAPER_BACKBONES = ("gpt-3.5-turbo-1106", "gpt-4-turbo", "orca2-13b", "llama3-8b")


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


def prepare_endpoint(api_base: str | None) -> None:
    """Keep localhost model traffic out of an inherited HTTP(S) proxy.

    Update only this process, preserve existing exclusions, and never bypass
    proxies for a remote endpoint. Both the SDK and DSPy honor NO_PROXY.
    """
    hostname = urlsplit(api_base or "").hostname
    if not hostname:
        return
    try:
        local = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        local = hostname.casefold() == "localhost"
    if not local:
        return
    exclusions = []
    for variable in ("NO_PROXY", "no_proxy"):
        exclusions.extend(x.strip() for x in os.getenv(variable, "").split(",") if x.strip())
    exclusions = list(dict.fromkeys([*exclusions, hostname]))
    for variable in ("NO_PROXY", "no_proxy"):
        os.environ[variable] = ",".join(exclusions)


@dataclass(frozen=True)
class ModelConfig:
    """OpenAI or an OpenAI-compatible endpoint; artifacts never contain a secret."""

    model: str = MODEL_NAME
    api_base: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not 0 <= self.temperature <= 2 or self.max_tokens <= 0:
            raise ValueError("Invalid temperature or max_tokens")
        if self.model in {"orca2-13b", "orca2-7b", "llama3-8b"} and not self.api_base:
            raise ValueError("An open-source backbone requires its --api-base endpoint")

    def api_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if key:
            return key
        if self.api_base:
            return "local"
        raise ValueError(f"Set {self.api_key_env}")

    def to_dict(self) -> dict:
        return asdict(self)


class OpenAILanguageModel:
    def __init__(
        self, api_key: str | None = None, *, config: ModelConfig | None = None,
    ) -> None:
        from openai import OpenAI

        self.config = config or ModelConfig()
        prepare_endpoint(self.config.api_base)
        self.client = OpenAI(
            api_key=api_key or self.config.api_key(), base_url=self.config.api_base,
            timeout=120.0, max_retries=2,
        )
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        response = self.client.chat.completions.create(
            model=self.config.model, temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.usage:
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        if response.choices[0].finish_reason == "length":
            raise RuntimeError("Model output was truncated; increase --max-tokens")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The model returned an empty response")
        return content
