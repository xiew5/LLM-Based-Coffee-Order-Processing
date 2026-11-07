from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .utils import get_logger, write_json


@dataclass
class GemmaAPIConfig:
    api_key_env: str = "GEMMA_API_KEY"
    model_name: str = "qwen2.5-coder:7b-instruct"
    api_provider: str = "ollama"
    api_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    requests_per_minute: int = 30
    endpoint_template: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GemmaAPIClient:
    """
    Adapter for model API calls.

    NOTE:
    - This class is intentionally isolated so provider-specific request wiring is in one place.
    - Supports Ollama local API, Google Generative Language style, and OpenAI-compatible style.
    - If your provider differs, update `_build_request` and `_extract_text`.
    """

    def __init__(self, config: GemmaAPIConfig, raw_response_dir: Path):
        self.config = config
        self.raw_response_dir = raw_response_dir
        self.logger = get_logger()
        self.api_key = os.getenv(config.api_key_env, "").strip()
        self._last_request_at: Optional[float] = None

    def _min_interval_seconds(self) -> float:
        rpm = max(1, int(self.config.requests_per_minute))
        return 60.0 / rpm

    def _throttle_if_needed(self) -> None:
        now = time.monotonic()
        if self._last_request_at is None:
            self._last_request_at = now
            return

        elapsed = now - self._last_request_at
        wait_time = self._min_interval_seconds() - elapsed
        if wait_time > 0:
            time.sleep(wait_time)

        self._last_request_at = time.monotonic()

    def _extract_retry_after_seconds(self, response: requests.Response) -> Optional[float]:
        retry_header = response.headers.get("Retry-After")
        if retry_header:
            try:
                return float(retry_header)
            except ValueError:
                pass

        match = re.search(r"retry in\s+([\d.]+)s", response.text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _build_url(self) -> str:
        provider = self.config.api_provider.lower()
        if provider == "google_genai":
            return self.config.endpoint_template.format(model=self.config.model_name)

        if provider == "ollama":
            base = self.config.api_base_url.rstrip("/")
            return f"{base}/api/chat"

        base = self.config.api_base_url.rstrip("/")
        return f"{base}/chat/completions"

    def _build_request(self, prompt_bundle: Dict[str, str]) -> Dict[str, Any]:
        """
        Provider-specific payload builder.
        """
        provider = self.config.api_provider.lower()

        if provider == "google_genai":
            combined_text = (
                f"System:\n{prompt_bundle['system_prompt']}\n\n"
                f"User:\n{prompt_bundle['user_prompt']}"
            )
            payload: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": combined_text}]}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens,
                },
            }
            return payload

        if provider == "ollama":
            return {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": prompt_bundle["system_prompt"]},
                    {"role": "user", "content": prompt_bundle["user_prompt"]},
                ],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }

        if provider in {"qwen_openai", "openai_compatible"}:
            return {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": prompt_bundle["system_prompt"]},
                    {"role": "user", "content": prompt_bundle["user_prompt"]},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }

        raise ValueError(f"Unsupported api_provider: {self.config.api_provider}")

    def _build_headers(self) -> Dict[str, str]:
        provider = self.config.api_provider.lower()
        if provider in {"google_genai", "ollama"}:
            return {"Content-Type": "application/json"}

        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_query_params(self) -> Dict[str, str]:
        provider = self.config.api_provider.lower()
        if provider == "google_genai":
            return {"key": self.api_key}
        return {}

    def _extract_text(self, response_json: Dict[str, Any]) -> str:
        provider = self.config.api_provider.lower()

        if provider == "ollama":
            message = response_json.get("message", {})
            content = message.get("content", "")
            return str(content)

        if provider == "google_genai":
            candidates = response_json.get("candidates", [])
            if not candidates:
                return ""

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                return ""

            text = parts[0].get("text", "")
            return str(text)

        choices = response_json.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content

        # Fallback for some providers
        return str(choices[0].get("text", ""))

    def generate(self, prompt_bundle: Dict[str, str], request_id: str) -> str:
        provider = self.config.api_provider.lower()
        requires_api_key = provider in {"qwen_openai", "openai_compatible", "google_genai"}

        if requires_api_key and not self.api_key:
            raise RuntimeError(
                f"Missing API key env var '{self.config.api_key_env}'. "
                "Set it in your environment before running model inference."
            )

        url = self._build_url()
        payload = self._build_request(prompt_bundle)
        headers = self._build_headers()
        query_params = self._build_query_params()

        last_error: Optional[str] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                if provider != "ollama":
                    self._throttle_if_needed()
                response = requests.post(
                    url,
                    params=query_params,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.config.timeout_seconds,
                )

                raw_json: Dict[str, Any]
                try:
                    raw_json = response.json()
                except Exception:  # noqa: BLE001
                    raw_json = {"non_json_response": response.text}

                write_json(self.raw_response_dir / f"{request_id}.json", raw_json)

                if response.status_code >= 500 or response.status_code == 429:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    self.logger.warning(
                        "Transient API error on attempt %d/%d: %s",
                        attempt,
                        self.config.max_retries,
                        last_error,
                    )
                    if attempt < self.config.max_retries:
                        retry_after = self._extract_retry_after_seconds(response)
                        backoff_wait = self.config.retry_backoff_seconds * attempt
                        sleep_for = max(backoff_wait, retry_after or 0.0)
                        time.sleep(sleep_for)
                    continue

                if response.status_code >= 400:
                    raise RuntimeError(f"API request failed: HTTP {response.status_code}: {response.text[:1000]}")

                text = self._extract_text(raw_json)
                if not text:
                    raise RuntimeError("API response had no text content in expected location.")

                return text
            except requests.RequestException as exc:
                last_error = str(exc)
                self.logger.warning(
                    "Request error on attempt %d/%d: %s",
                    attempt,
                    self.config.max_retries,
                    last_error,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * attempt)

        raise RuntimeError(f"API request failed after retries. Last error: {last_error}")
