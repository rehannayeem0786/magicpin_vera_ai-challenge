"""
Vera Pro — Shared State Store (serverless-safe)
================================================
Problem: on Vercel Fluid Compute, parallel judge requests can land on
DIFFERENT function instances. Purely in-memory state then fragments —
a tick may run with contexts it never saw, and mid-test context
injections can be missed entirely (scored "ignores new context").

Solution: keep the in-memory dicts as the working cache (identical
behavior + tests), but back them with a single shared JSON blob in
Upstash Redis (REST) when UPSTASH_REDIS_REST_URL is configured:

  - every request:  remote load -> apply onto local cache
  - after mutation: merge local cache into remote (version-aware for
    contexts, last-writer-wins for conversations/anti-repetition)
  - suppression dedup: atomic SETNX claims — parallel instances can
    never double-send the same trigger
  - teardown: wipes the remote blob + all claim keys

If Redis is not configured or errors, everything degrades gracefully
to plain in-memory behavior (never crashes an endpoint).
"""

import hashlib
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("vera_pro")

REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

STATE_KEY = "vera:state:v1"
CLAIM_PREFIX = "vera:sup:"
CLAIM_TTL_S = 2 * 60 * 60  # suppression claims live 2h max
REDIS_TIMEOUT_S = 3.0

_ENABLED = bool(REDIS_URL and REDIS_TOKEN)


def is_remote() -> bool:
    """True when a shared Redis backend is configured."""
    return _ENABLED


# ─────────────────────────────────────────────────────────────────────
# Low-level REST commands
# ─────────────────────────────────────────────────────────────────────

async def _cmd(*args: Any) -> Any:
    """Execute one Redis command via Upstash REST. Returns result or None."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(REDIS_TIMEOUT_S)) as client:
        resp = await client.post(
            REDIS_URL,
            json=list(args),
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"},
        )
    if resp.status_code != 200:
        logger.warning(f"Redis command failed: {resp.status_code} {resp.text[:120]}")
        return None
    return resp.json().get("result")


async def _pipeline(commands: list[list[Any]]) -> list[Any] | None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(REDIS_TIMEOUT_S)) as client:
        resp = await client.post(
            f"{REDIS_URL}/pipeline",
            json=commands,
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"},
        )
    if resp.status_code != 200:
        logger.warning(f"Redis pipeline failed: {resp.status_code} {resp.text[:120]}")
        return None
    return [item.get("result") for item in resp.json()]


# ─────────────────────────────────────────────────────────────────────
# State blob load/save (with merge safety)
# ─────────────────────────────────────────────────────────────────────

def _claim_key(suppression_key: str) -> str:
    digest = hashlib.sha1(suppression_key.encode("utf-8")).hexdigest()
    return f"{CLAIM_PREFIX}{digest}"


async def load_blob() -> dict | None:
    """Fetch the shared state blob. None on any failure / not configured."""
    if not _ENABLED:
        return None
    try:
        result = await _cmd("GET", STATE_KEY)
        if not result:
            return {}
        return json.loads(result)
    except Exception as e:
        logger.warning(f"Redis load failed ({type(e).__name__}); using local state")
        return None


async def save_blob(local_blob: dict) -> None:
    """
    Merge-save: re-read remote, keep the higher version per context key,
    let the local writer win for conversations / sent_bodies it touched.
    Prevents a long-running tick from clobbering a context push that
    landed on another instance mid-composition.
    """
    if not _ENABLED:
        return
    try:
        remote = await load_blob() or {}
        merged = _merge(remote, local_blob)
        await _cmd("SET", STATE_KEY, json.dumps(merged, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"Redis save failed ({type(e).__name__}); state kept locally")


def _merge(remote: dict, local: dict) -> dict:
    merged = dict(remote)

    # Contexts: highest version wins per (scope|id)
    r_ctx = dict(merged.get("contexts", {}))
    for key, entry in local.get("contexts", {}).items():
        existing = r_ctx.get(key)
        if not existing or int(entry.get("version", 0)) >= int(existing.get("version", 0)):
            r_ctx[key] = entry
    merged["contexts"] = r_ctx

    # Conversations: local writer wins per conversation
    r_conv = dict(merged.get("conversations", {}))
    r_conv.update(local.get("conversations", {}))
    merged["conversations"] = r_conv

    # Sent bodies: local writer wins per merchant
    r_bodies = dict(merged.get("sent_bodies", {}))
    r_bodies.update(local.get("sent_bodies", {}))
    merged["sent_bodies"] = r_bodies

    # Suppression registry: union (claims are the atomic source of truth;
    # this list is for observability/teardown counts)
    r_sup = set(merged.get("suppression", [])) | set(local.get("suppression", []))
    merged["suppression"] = sorted(r_sup)

    return merged


async def claim_suppression(suppression_key: str) -> bool:
    """
    Atomically claim a suppression key. True → we may send.
    Uses SET NX so two parallel instances can never both send.
    Always True in memory-only mode (caller keeps its own set).
    """
    if not _ENABLED:
        return True
    try:
        result = await _cmd(
            "SET", _claim_key(suppression_key), "1", "NX", "EX", CLAIM_TTL_S
        )
        return result == "OK"
    except Exception as e:
        logger.warning(f"Redis claim failed ({type(e).__name__}); allowing send")
        return True


async def release_suppression(suppression_key: str) -> None:
    """Undo a claim (composition failed — let a future tick retry)."""
    if not _ENABLED:
        return
    try:
        await _cmd("DEL", _claim_key(suppression_key))
    except Exception as e:
        logger.warning(f"Redis release failed: {type(e).__name__}")


async def wipe_remote() -> int:
    """Delete the state blob and every claim key. Returns keys removed."""
    if not _ENABLED:
        return 0
    try:
        keys = await _cmd("KEYS", f"{CLAIM_PREFIX}*") or []
        commands = [["DEL", STATE_KEY]] + [["DEL", k] for k in keys]
        results = await _pipeline(commands)
        return sum(1 for r in (results or []) if r and r > 0)
    except Exception as e:
        logger.warning(f"Redis wipe failed: {type(e).__name__}")
        return 0

