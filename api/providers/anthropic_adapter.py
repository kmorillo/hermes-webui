"""Anthropic API adapter for Hermes Web UI.

Proxies requests to https://api.anthropic.com using only stdlib
(urllib.request, http.client, json, threading, queue, time, logging, os,
random, string) — no third-party HTTP libraries.

Return envelope for non-streaming functions:
  {"ok": bool, "data": ..., "error": str|None, "status_code": int|None}

Key detection:
  Standard key:  settings["anthropic_api_key"] or env ANTHROPIC_API_KEY
  Admin key:     settings["anthropic_admin_key"] or env ANTHROPIC_ADMIN_KEY
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import queue
import random
import string
import threading
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_ANTHROPIC_BASE = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
_FILES_BETA_HEADER = "files-api-2025-04-14"
_API_TIMEOUT = 30.0
_STREAM_READ_TIMEOUT = 120.0
_BATCH_POLL_TIMEOUT = 10.0


# ── Key helpers ───────────────────────────────────────────────────────────────

def _get_key(settings: dict | None, kind: str = "standard") -> str:
    if not settings:
        settings = {}
    if kind == "admin":
        return (
            settings.get("anthropic_admin_key", "").strip()
            or os.getenv("ANTHROPIC_ADMIN_KEY", "").strip()
            or ""
        )
    return (
        settings.get("anthropic_api_key", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
        or ""
    )


def _is_admin_key(key: str) -> bool:
    return key.startswith("sk-ant-admin")


def _make_headers(
    api_key: str,
    *,
    beta: str | None = None,
    content_type: str = "application/json",
) -> dict:
    h = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": content_type,
        "accept": "application/json",
    }
    if beta:
        h["anthropic-beta"] = beta
    return h


# ── Error helpers ─────────────────────────────────────────────────────────────

def _map_error(status: int, payload: dict) -> str:
    err_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
    err_type = err_obj.get("type", "") if isinstance(err_obj, dict) else ""
    err_msg = err_obj.get("message", str(payload)) if isinstance(err_obj, dict) else str(payload)
    if status in (401, 403) or err_type in ("authentication_error", "permission_error"):
        return f"invalid_key: {err_msg}"
    if status == 529 or err_type == "overloaded_error":
        return f"overloaded: {err_msg}"
    if status == 429:
        return f"rate_limited: {err_msg}"
    return err_msg or f"HTTP {status}"


def _ok(data) -> dict:
    return {"ok": True, "data": data, "error": None, "status_code": 200}


def _err(msg: str, status: int | None = None) -> dict:
    return {"ok": False, "data": None, "error": msg, "status_code": status}


# ── Request helper ────────────────────────────────────────────────────────────

def _request(
    method: str,
    path: str,
    api_key: str,
    *,
    body: dict | None = None,
    beta: str | None = None,
    timeout: float = _API_TIMEOUT,
) -> dict:
    url = _ANTHROPIC_BASE + path
    headers = _make_headers(api_key, beta=beta)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            parsed = json.loads(raw) if raw else {}
            return _ok(parsed)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
        return _err(_map_error(exc.code, payload), exc.code)
    except urllib.error.URLError as exc:
        return _err(f"network_error: {exc.reason}")
    except Exception as exc:
        return _err(str(exc))


# ── Models ────────────────────────────────────────────────────────────────────

def list_models(settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    return _request("GET", "/v1/models", key)


# ── Count tokens ──────────────────────────────────────────────────────────────

def count_tokens(
    api_key_or_settings,
    model: str | None = None,
    messages: list | None = None,
    *,
    system: str | None = None,
) -> dict:
    if isinstance(api_key_or_settings, dict):
        key = _get_key(api_key_or_settings)
        body: dict = {}
        if model:
            body["model"] = model
        if messages is not None:
            body["messages"] = messages
        if system is not None:
            body["system"] = system
    else:
        key = api_key_or_settings
        body = {"model": model, "messages": messages or []}
        if system is not None:
            body["system"] = system
    if not key:
        return _err("no_key", 401)
    return _request("POST", "/v1/messages/count_tokens", key, body=body)


# ── Messages ──────────────────────────────────────────────────────────────────

def send_message(payload: dict, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    return _request("POST", "/v1/messages", key, body=payload, timeout=_API_TIMEOUT)


# Keep backward-compat alias used by existing routes.py imports
create_message = send_message


# ── Streaming messages ────────────────────────────────────────────────────────

def stream_message(
    api_key: str,
    model: str,
    messages: list,
    out_queue: queue.Queue,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
) -> None:
    """Runs in a daemon thread.  Streams /v1/messages and puts event dicts onto out_queue.

    Event shapes put on the queue:
      {"event": "claude_start", "data": {usage info}}
      {"event": "claude_delta", "data": {"text": chunk}}
      {"event": "claude_done",  "data": {stop_reason, usage}}
      {"event": "claude_error", "data": {"message": "..."}}
    """
    parsed_url = urllib.parse.urlparse(_ANTHROPIC_BASE)
    host = parsed_url.hostname
    port = parsed_url.port or 443

    body_dict: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if system:
        body_dict["system"] = system

    body_bytes = json.dumps(body_dict).encode("utf-8")
    headers = _make_headers(api_key)
    headers["content-length"] = str(len(body_bytes))

    conn: http.client.HTTPSConnection | None = None
    try:
        conn = http.client.HTTPSConnection(host, port, timeout=_STREAM_READ_TIMEOUT)
        conn.request("POST", "/v1/messages", body=body_bytes, headers=headers)
        resp = conn.getresponse()

        if resp.status != 200:
            raw = resp.read()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
            out_queue.put({"event": "claude_error", "data": {"message": _map_error(resp.status, payload)}})
            return

        while True:
            line_bytes = resp.fp.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                ev = json.loads(data_str)
            except Exception:
                continue
            ev_type = ev.get("type", "")
            if ev_type == "message_start":
                usage = ev.get("message", {}).get("usage", {})
                out_queue.put({"event": "claude_start", "data": usage})
            elif ev_type == "content_block_delta":
                delta = ev.get("delta", {})
                if delta.get("type") == "text_delta":
                    out_queue.put({"event": "claude_delta", "data": {"text": delta.get("text", "")}})
            elif ev_type == "message_delta":
                delta = ev.get("delta", {})
                usage = ev.get("usage", {})
                out_queue.put({"event": "claude_done", "data": {
                    "stop_reason": delta.get("stop_reason"),
                    "output_tokens": usage.get("output_tokens"),
                }})
                return
            elif ev_type == "error":
                err_obj = ev.get("error", {})
                out_queue.put({"event": "claude_error", "data": {"message": err_obj.get("message", str(ev))}})
                return

        out_queue.put({"event": "claude_done", "data": {}})
    except Exception as exc:
        out_queue.put({"event": "claude_error", "data": {"message": str(exc)}})
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ── Batches ───────────────────────────────────────────────────────────────────

def list_batches(settings: dict | None = None, *, before_id: str | None = None, limit: int = 20) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    qs = f"?limit={limit}"
    if before_id:
        qs += f"&before_id={urllib.parse.quote(before_id)}"
    return _request("GET", f"/v1/messages/batches{qs}", key)


def create_batch(payload: dict, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    return _request("POST", "/v1/messages/batches", key, body=payload)


def get_batch(batch_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    return _request("GET", f"/v1/messages/batches/{urllib.parse.quote(batch_id)}", key)


def get_batch_results(batch_id: str, settings: dict | None = None) -> dict:
    """Fetch batch results (JSONL format) and return as list."""
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    url = _ANTHROPIC_BASE + f"/v1/messages/batches/{urllib.parse.quote(batch_id)}/results"
    headers = _make_headers(key)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        results = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
        return _ok(results)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
        return _err(_map_error(exc.code, payload), exc.code)
    except Exception as exc:
        return _err(str(exc))


def cancel_batch(batch_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    return _request("POST", f"/v1/messages/batches/{urllib.parse.quote(batch_id)}/cancel", key, body={})


# ── Files API ─────────────────────────────────────────────────────────────────

def list_files(settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    return _request("GET", "/v1/files", key, beta=_FILES_BETA_HEADER)


def get_file_metadata(file_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    return _request("GET", f"/v1/files/{urllib.parse.quote(file_id)}", key, beta=_FILES_BETA_HEADER)


def delete_file(file_id: str, settings: dict | None = None) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    url = _ANTHROPIC_BASE + f"/v1/files/{urllib.parse.quote(file_id)}"
    headers = _make_headers(key, beta=_FILES_BETA_HEADER)
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
            raw = resp.read()
            try:
                return _ok(json.loads(raw)) if raw else _ok({"deleted": True})
            except Exception:
                return _ok({"deleted": True})
    except urllib.error.HTTPError as exc:
        if exc.code == 204:
            return _ok({"deleted": True})
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
        return _err(_map_error(exc.code, payload), exc.code)
    except Exception as exc:
        return _err(str(exc))


def upload_file(
    filename: str,
    content: bytes,
    mime_type: str,
    settings: dict | None = None,
) -> dict:
    key = _get_key(settings)
    if not key:
        return _err("no_key", 401)
    boundary = "".join(random.choices(string.ascii_letters + string.digits, k=28))
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {
        "x-api-key": key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "anthropic-beta": _FILES_BETA_HEADER,
        "content-type": f"multipart/form-data; boundary={boundary}",
        "content-length": str(len(body)),
    }
    url = _ANTHROPIC_BASE + "/v1/files"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
            raw = resp.read()
            return _ok(json.loads(raw)) if raw else _ok({})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"error": {"message": raw.decode("utf-8", errors="replace")}}
        return _err(_map_error(exc.code, payload), exc.code)
    except Exception as exc:
        return _err(str(exc))


# ── Usage & Cost ──────────────────────────────────────────────────────────────

def get_usage(
    settings: dict | None = None,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    granularity: str = "day",
    group_by: str | None = None,
) -> dict:
    key = _get_key(settings, kind="admin")
    if not key:
        return _err("no_admin_key", 401)
    if not _is_admin_key(key):
        return _err("admin_key_required: provide an sk-ant-admin key for usage data", 403)
    qs_parts = [f"granularity={urllib.parse.quote(granularity)}"]
    if start_time:
        qs_parts.append(f"start_time={urllib.parse.quote(start_time)}")
    if end_time:
        qs_parts.append(f"end_time={urllib.parse.quote(end_time)}")
    if group_by:
        qs_parts.append(f"group_by={urllib.parse.quote(group_by)}")
    path = "/v1/organizations/usage?" + "&".join(qs_parts)
    return _request("GET", path, key)


def get_claude_code_analytics(
    settings: dict | None = None,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    key = _get_key(settings, kind="admin")
    if not key:
        return _err("no_admin_key", 401)
    if not _is_admin_key(key):
        return _err("admin_key_required: provide an sk-ant-admin key for analytics", 403)
    qs_parts = []
    if start_time:
        qs_parts.append(f"start_time={urllib.parse.quote(start_time)}")
    if end_time:
        qs_parts.append(f"end_time={urllib.parse.quote(end_time)}")
    path = "/v1/organizations/usage_report/claude_code"
    if qs_parts:
        path += "?" + "&".join(qs_parts)
    return _request("GET", path, key)


# ── Admin — Users & Orgs ──────────────────────────────────────────────────────

def list_users(
    settings: dict | None = None,
    *,
    limit: int = 100,
    after_id: str | None = None,
) -> dict:
    key = _get_key(settings, kind="admin")
    if not key:
        return _err("no_admin_key", 401)
    if not _is_admin_key(key):
        return _err("admin_key_required: provide an sk-ant-admin key for admin endpoints", 403)
    qs = f"?limit={limit}"
    if after_id:
        qs += f"&after_id={urllib.parse.quote(after_id)}"
    return _request("GET", f"/v1/admin/users{qs}", key)


def get_organization(settings: dict | None = None) -> dict:
    key = _get_key(settings, kind="admin")
    if not key:
        return _err("no_admin_key", 401)
    if not _is_admin_key(key):
        return _err("admin_key_required: provide an sk-ant-admin key for admin endpoints", 403)
    return _request("GET", "/v1/admin/organizations/me", key)
