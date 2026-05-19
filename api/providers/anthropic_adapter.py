"""Anthropic API adapter for Hermes Web UI.

Proxies requests to https://api.anthropic.com using only the stdlib
urllib.request — no third-party HTTP libraries required.

Key detection:
  - Standard key:  settings["anthropic_api_key"] or env ANTHROPIC_API_KEY
  - Admin key:     settings["anthropic_admin_key"] or env ANTHROPIC_ADMIN_KEY
    (detected by prefix "sk-ant-admin")

Anthropic API details:
  - Base URL:       https://api.anthropic.com
  - Version header: anthropic-version: 2023-06-01
  - Files beta:     anthropic-beta: files-api-2025-04-14
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"
_FILES_BETA = "files-api-2025-04-14"
_TIMEOUT = 60


def _get_key(settings: dict | None, kind: str = "standard") -> str:
    """Return the API key for the requested kind ('standard' or 'admin')."""
    if settings is None:
        settings = {}
    if kind == "admin":
        key = (
            settings.get("anthropic_admin_key", "").strip()
            or os.getenv("ANTHROPIC_ADMIN_KEY", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "").strip()
        )
    else:
        key = (
            settings.get("anthropic_api_key", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "").strip()
        )
    return key


def _is_admin_key(key: str) -> bool:
    return key.startswith("sk-ant-admin")


def _build_headers(api_key: str, extra_beta: str | None = None) -> dict:
    hdrs = {
        "x-api-key": api_key,
        "anthropic-version": _API_VERSION,
        "content-type": "application/json",
    }
    if extra_beta:
        hdrs["anthropic-beta"] = extra_beta
    return hdrs


def _call(
    method: str,
    path: str,
    api_key: str,
    *,
    body: dict | None = None,
    beta: str | None = None,
    timeout: int = _TIMEOUT,
) -> dict:
    """Make a request to the Anthropic API and return the parsed JSON response."""
    url = _BASE_URL + path
    headers = _build_headers(api_key, beta)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
        status = exc.code
        raise AnthropicError(status, payload) from exc


class AnthropicError(Exception):
    def __init__(self, status: int, payload: dict):
        self.status = status
        self.payload = payload
        super().__init__(f"Anthropic API error {status}: {payload}")


def list_models(settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key", "models": []}
    try:
        return _call("GET", "/v1/models", key)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("list_models failed: %s", e)
        return {"error": str(e)}


def create_message(payload: dict, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    try:
        return _call("POST", "/v1/messages", key, body=payload)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("create_message failed: %s", e)
        return {"error": str(e)}


def list_batches(settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key", "data": []}
    try:
        return _call("GET", "/v1/messages/batches?limit=100", key)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("list_batches failed: %s", e)
        return {"error": str(e)}


def create_batch(payload: dict, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    try:
        return _call("POST", "/v1/messages/batches", key, body=payload)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("create_batch failed: %s", e)
        return {"error": str(e)}


def get_batch(batch_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    try:
        return _call("GET", f"/v1/messages/batches/{batch_id}", key)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("get_batch failed: %s", e)
        return {"error": str(e)}


def cancel_batch(batch_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    try:
        return _call("POST", f"/v1/messages/batches/{batch_id}/cancel", key)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("cancel_batch failed: %s", e)
        return {"error": str(e)}


def get_batch_results(batch_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    try:
        return _call("GET", f"/v1/messages/batches/{batch_id}/results", key)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("get_batch_results failed: %s", e)
        return {"error": str(e)}


def list_files(settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key", "data": []}
    try:
        return _call("GET", "/v1/files", key, beta=_FILES_BETA)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("list_files failed: %s", e)
        return {"error": str(e)}


def get_file_metadata(file_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    try:
        return _call("GET", f"/v1/files/{file_id}", key, beta=_FILES_BETA)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("get_file_metadata failed: %s", e)
        return {"error": str(e)}


def delete_file(file_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    try:
        return _call("DELETE", f"/v1/files/{file_id}", key, beta=_FILES_BETA)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("delete_file failed: %s", e)
        return {"error": str(e)}


def upload_file(
    filename: str,
    content: bytes,
    mime_type: str,
    settings: dict | None = None,
) -> dict:
    """Upload a file to the Anthropic Files API using multipart/form-data."""
    key = _get_key(settings)
    if not key:
        return {"error": "no_key"}
    import io
    boundary = "hermeswebui_boundary_8x7y"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    headers = {
        "x-api-key": key,
        "anthropic-version": _API_VERSION,
        "anthropic-beta": _FILES_BETA,
        "content-type": f"multipart/form-data; boundary={boundary}",
    }
    url = _BASE_URL + "/v1/files"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
        return {"error": payload, "status": exc.code}
    except Exception as e:
        logger.warning("upload_file failed: %s", e)
        return {"error": str(e)}


def get_usage(settings: dict | None = None) -> dict:
    """Get usage data — requires an admin API key."""
    key = _get_key(settings, kind="admin")
    if not key:
        return {"error": "no_admin_key"}
    if not _is_admin_key(key):
        return {"error": "admin_key_required", "message": "Provide an sk-ant-admin key for usage data."}
    try:
        return _call("GET", "/v1/organizations/usage?limit=100", key)
    except AnthropicError as e:
        return {"error": e.payload, "status": e.status}
    except Exception as e:
        logger.warning("get_usage failed: %s", e)
        return {"error": str(e)}
