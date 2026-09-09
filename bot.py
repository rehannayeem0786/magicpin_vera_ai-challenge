"""
Vera Pro — FastAPI Bot Server
================================
magicpin AI Challenge submission.
Exposes 5 endpoints per the testing brief:
  POST /v1/context   — receive context pushes
  POST /v1/tick      — periodic wake-up; bot initiates conversations
  POST /v1/reply     — handle merchant/customer replies
  GET  /v1/healthz   — liveness probe
  GET  /v1/metadata  — bot identity

Architecture:
  Mistral LLM chain (medium primary → large → nemo) for composition
  Trigger-specific prompt routing for 15+ trigger kinds
  Conversation state machine for multi-turn handling
  Post-composition validation for quality assurance

Author: Vera Pro Team — magicpin AI Challenge
"""

import os
import time
import uuid
import json
import logging
import asyncio
from collections import deque
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env before importing modules that read env vars
load_dotenv()

from composer import EngagementComposer
from conversation_handlers import ConversationManager
from validators import detect_auto_reply, detect_intent, normalize_text

# ─────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vera_pro")

# ─────────────────────────────────────────────────────────────────────
# App + global state
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vera Pro — magicpin AI Challenge Bot",
    version="1.0.0",
    description="AI-powered merchant engagement assistant",
)

START_TIME = time.time()

# In-memory stores
contexts: dict[tuple[str, str], dict] = {}     # (scope, context_id) -> {version, payload}
conversations_mgr = ConversationManager()       # Conversation state machine
composer = EngagementComposer()                 # LLM composition engine

# Track which triggers have been used (for suppression/dedup)
used_suppression_keys: set[str] = set()

# Per-merchant memory of recently sent bodies (anti-repetition across conversations)
sent_bodies: dict[str, deque] = {}

# Import primary model name for metadata reporting
from composer import PRIMARY_MODEL

# Shared-state bridge: on Vercel Fluid Compute, parallel judge requests can
# land on different instances. When Upstash Redis is configured, every
# request hydrates from / merges into one shared blob so no instance ever
# runs with stale or missing contexts. Without Redis → memory-only (local).
import state_store


def _export_blob() -> dict:
    """Snapshot the in-memory working state for the shared blob."""
    return {
        "contexts": {
            f"{scope}|{cid}": entry for (scope, cid), entry in contexts.items()
        },
        "conversations": conversations_mgr.export(),
        "sent_bodies": {mid: list(dq) for mid, dq in sent_bodies.items()},
        "suppression": sorted(used_suppression_keys),
    }


def _apply_blob(blob: dict) -> None:
    """Merge a shared blob onto the local working cache."""
    if not blob:
        return
    for key, entry in blob.get("contexts", {}).items():
        scope, _, cid = key.rpartition("|")
        if not scope or not cid:
            continue
        tuple_key = (scope, cid)
        existing = contexts.get(tuple_key)
        if not existing or int(entry.get("version", 0)) >= int(existing["version"]):
            contexts[tuple_key] = entry
    conversations_mgr.import_state(blob.get("conversations", {}))
    for mid, bodies in blob.get("sent_bodies", {}).items():
        dq = sent_bodies.setdefault(mid, deque(maxlen=5))
        for b in bodies:
            if b not in dq:
                dq.append(b)
    used_suppression_keys.update(blob.get("suppression", []))


async def _hydrate_state() -> None:
    """Pull shared state onto the local cache before processing a request."""
    if not state_store.is_remote():
        return
    blob = await state_store.load_blob()
    if blob:
        _apply_blob(blob)


async def _persist_state() -> None:
    """Merge local mutations back into the shared blob (version-safe)."""
    if not state_store.is_remote():
        return
    await state_store.save_blob(_export_blob())

# ─────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────

class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────

def _get_context(scope: str, context_id: str) -> dict | None:
    """Get a stored context payload."""
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def _get_category_for_merchant(merchant: dict) -> dict | None:
    """Look up the category context for a merchant."""
    cat_slug = merchant.get("category_slug", "")
    return _get_context("category", cat_slug)


def _count_contexts() -> dict[str, int]:
    """Count stored contexts by scope."""
    counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    """Liveness probe — judge polls every 60s."""
    await _hydrate_state()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": _count_contexts(),
    }


@app.get("/v1/metadata")
async def metadata():
    """Bot identity and approach."""
    return {
        "team_name": "Vera Pro",
        "team_members": ["AI Challenge Participant"],
        "model": f"{PRIMARY_MODEL} via Mistral API (fallback chain: mistral-large → mistral-nemo)",
        "approach": "4-context composition framework with trigger-specific prompt routing, "
                    "specificity-anchor anti-hallucination engine, validate→LLM-repair→"
                    "deterministic-repair pipeline, conversation state machine for multi-turn, "
                    "auto-reply detection with ≤2-turn exit, intent transition handling, "
                    "Hindi-English code-mixing",
        "contact_email": "participant@challenge.com",
        "version": "2.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/teardown")
async def teardown():
    """
    End-of-test state wipe (testing brief §11 privacy rule).
    Wipes all stored contexts, conversations, and dedup state.
    """
    n_contexts = len(contexts)
    n_convs = len(conversations_mgr.conversations)
    contexts.clear()
    conversations_mgr.conversations.clear()
    used_suppression_keys.clear()
    sent_bodies.clear()
    remote_keys = await state_store.wipe_remote()
    logger.info(f"TEARDOWN: wiped {n_contexts} contexts, {n_convs} conversations, "
                f"{remote_keys} remote keys")
    return {
        "wiped": True,
        "contexts_removed": n_contexts,
        "conversations_removed": n_convs,
        "remote_keys_wiped": remote_keys,
        "wiped_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextPush):
    """
    Receive a context push from the judge.
    Idempotent by (context_id, version). Higher version replaces atomically.
    """
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope", "details": f"Scope must be one of {valid_scopes}"}

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    # Version check — idempotent
    if current and current["version"] >= body.version:
        if current["version"] == body.version:
            # Same version — idempotent, accept silently
            return {
                "accepted": True,
                "ack_id": f"ack_{body.context_id}_v{body.version}",
                "stored_at": datetime.utcnow().isoformat() + "Z",
            }
        else:
            return {
                "accepted": False,
                "reason": "stale_version",
                "current_version": current["version"],
            }

    # Store the new context
    contexts[key] = {
        "version": body.version,
        "payload": body.payload,
        "delivered_at": body.delivered_at,
    }

    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")
    await _persist_state()

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(body: TickRequest):
    """
    Periodic wake-up. Bot inspects context state and decides
    whether to send proactive messages.

    Robustness design:
    - Candidates resolved synchronously (fast dict lookups)
    - One trigger per merchant per tick (highest urgency wins) — avoids spam
    - Compositions run concurrently under a shared 26s deadline so the
      endpoint always answers within the 30s contract
    """
    started = time.monotonic()
    actions = []
    await _hydrate_state()

    # ── Phase 1: resolve candidates (no LLM) ──
    candidates = []
    seen_merchants: set[str] = set()

    # Sort triggers by urgency desc so the best trigger per merchant wins
    def _urgency(trig_id: str) -> int:
        t = _get_context("trigger", trig_id) or {}
        return int(t.get("urgency", 1))

    for trig_id in sorted(body.available_triggers, key=_urgency, reverse=True):
        if time.monotonic() - started > 2.0:
            break
        trigger = _get_context("trigger", trig_id)
        if not trigger:
            logger.warning(f"Trigger {trig_id} not found in context store")
            continue

        suppression_key = trigger.get("suppression_key", "")
        if suppression_key and suppression_key in used_suppression_keys:
            logger.info(f"Trigger {trig_id} suppressed (key: {suppression_key})")
            continue

        merchant_id = trigger.get("merchant_id", "")
        merchant = _get_context("merchant", merchant_id)
        if not merchant:
            logger.warning(f"Merchant {merchant_id} not found for trigger {trig_id}")
            continue

        category = _get_category_for_merchant(merchant)
        if not category:
            logger.warning(f"Category {merchant.get('category_slug', '?')} not found for {merchant_id}")
            continue

        # One action per merchant per tick — highest-urgency trigger wins
        if merchant_id in seen_merchants:
            logger.info(f"Skipping {trig_id}: already messaging {merchant_id} this tick")
            continue
        seen_merchants.add(merchant_id)

        customer = None
        customer_id = trigger.get("customer_id")
        if customer_id:
            customer = _get_context("customer", customer_id)

        candidates.append({
            "trig_id": trig_id, "trigger": trigger, "merchant": merchant,
            "category": category, "customer": customer,
            "customer_id": customer_id, "suppression_key": suppression_key,
        })

    # Cap concurrent compositions to protect rate limits; process in waves
    MAX_CONCURRENT = 4
    DEADLINE_S = 26.0

    async def _compose_one(cand: dict) -> dict | None:
        try:
            recent = list(sent_bodies.get(cand["candidate_merchant_id"], []))
            return await composer.compose(
                cand["category"], cand["merchant"], cand["trigger"],
                cand["customer"],
                recent_sent_bodies=recent,
            )
        except Exception as e:
            logger.error(f"Composition failed for {cand['trig_id']}: {e}")
            return None

    async def _guarded_compose(cand: dict) -> dict | None:
        remaining = DEADLINE_S - (time.monotonic() - started)
        if remaining < 6.0:
            logger.info(f"Deadline guard: skipping composition of {cand['trig_id']}")
            return None
        try:
            return await asyncio.wait_for(_compose_one(cand), timeout=remaining)
        except asyncio.TimeoutError:
            logger.error(f"Compose timed out for {cand['trig_id']}")
            return None

    # Stash merchant id where the inner fn can find it
    for c in candidates:
        c["candidate_merchant_id"] = c["merchant"].get("merchant_id", "")

    # ── Phase 2: compose concurrently in bounded waves ──
    results: list[dict | None] = []
    for i in range(0, len(candidates), MAX_CONCURRENT):
        wave = candidates[i:i + MAX_CONCURRENT]
        if time.monotonic() - started > DEADLINE_S - 6.0:
            logger.info(f"Tick deadline: skipping remaining {len(candidates) - i} triggers")
            break
        results.extend(await asyncio.gather(*(_guarded_compose(c) for c in wave)))

    # ── Phase 3: build actions from successful compositions ──
    for cand, result in zip(candidates, results):
        if not result or not result.get("body"):
            continue

        trig_id = cand["trig_id"]
        merchant_id = cand["candidate_merchant_id"]
        conv_id = f"conv_{merchant_id}_{trig_id}_{int(time.time())}"

        # Atomic cross-instance claim — parallel ticks can never double-send
        if cand["suppression_key"] and not await state_store.claim_suppression(
            cand["suppression_key"]
        ):
            logger.info(f"Suppression {cand['suppression_key']} already claimed "
                        f"by another instance")
            continue

        body_text = normalize_text(result["body"])
        template_params = _extract_template_params(body_text, cand["merchant"])

        conversations_mgr.get_or_create(conv_id, merchant_id, cand["customer_id"])
        conversations_mgr.record_bot_send(conv_id, body_text, trig_id)

        if cand["suppression_key"]:
            used_suppression_keys.add(cand["suppression_key"])

        # Remember what we sent (per-merchant anti-repetition memory)
        mid = merchant_id
        bucket = sent_bodies.setdefault(mid, deque(maxlen=5))
        bucket.append(body_text)

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": cand["customer_id"],
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trig_id,
            "template_name": f"vera_{cand['trigger'].get('kind', 'generic')}_v1",
            "template_params": template_params,
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get(
                "suppression_key", cand["suppression_key"]
            ),
            "rationale": result.get("rationale", "Composed from 4-context framework"),
        })

    logger.info(f"Tick: {len(actions)} actions from "
                f"{len(body.available_triggers)} triggers "
                f"in {time.monotonic() - started:.1f}s")
    await _persist_state()
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    """
    Handle a reply from the simulated merchant/customer.
    Must respond within 30 seconds.
    """
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or ""
    message = body.message
    await _hydrate_state()

    # Process through conversation state machine
    routing = conversations_mgr.process_incoming(conv_id, message)
    intent = routing["intent"]
    conv_state = routing.get("conv_state") or {}

    logger.info(f"Reply in {conv_id}: intent={intent}, state={conv_state.get('state', '?')}")

    # ── Fast-path decisions (no LLM needed) ──

    # Auto-reply: 2+ times → exit
    if routing["should_end"] and routing["auto_reply_detected"]:
        conversations_mgr.end_conversation(conv_id)
        await _persist_state()
        return {
            "action": "end",
            "rationale": f"Auto-reply detected {conv_state.get('auto_reply_count', 2)}+ times. Graceful exit.",
        }

    # Hostile / not interested → graceful exit
    if intent in ("hostile", "not_interested"):
        conversations_mgr.end_conversation(conv_id)
        merchant = _get_context("merchant", merchant_id) or {}
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        name = owner or merchant.get("identity", {}).get("name", "")

        if intent == "hostile":
            # Gender-neutral Hinglish close
            exit_body = f"Theek hai {name}. Jab bhi help chahiye ho, Vera yahin hai. Best wishes!"
        else:
            exit_body = f"Koi baat nahi {name}. Jab bhi zarurat ho, Vera yahin hai. Have a great day!"

        return {
            "action": "send",
            "body": exit_body,
            "cta": "none",
            "rationale": f"Merchant signaled {intent}. Graceful exit with warm close.",
        }

    # Too many turns → exit
    if routing["should_end"]:
        conversations_mgr.end_conversation(conv_id)
        await _persist_state()
        return {
            "action": "end",
            "rationale": "Conversation exceeded max turns. Graceful exit.",
        }

    # ── LLM-powered reply ──

    # Get merchant and category for context
    merchant = _get_context("merchant", merchant_id) or {}
    category = _get_category_for_merchant(merchant) or {}

    try:
        result = await composer.compose_reply(
            conv_state, message, category, merchant
        )
    except Exception as e:
        logger.error(f"Reply composition failed: {e}")
        # Fallback based on intent
        result = _fallback_reply(intent, merchant)

    action = result.get("action", "send")

    # Record bot's response
    if action == "send" and result.get("body"):
        result["body"] = normalize_text(result["body"])
        conversations_mgr.record_bot_send(conv_id, result["body"])
    await _persist_state()

    if action == "end":
        conversations_mgr.end_conversation(conv_id)

    response = {"action": action}
    if action == "send":
        response["body"] = result.get("body", "")
        response["cta"] = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 1800)

    response["rationale"] = result.get("rationale", f"Reply based on detected intent: {intent}")

    return response


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def _extract_template_params(body: str, merchant: dict) -> list[str]:
    """Extract template parameters from a composed message body."""
    name = merchant.get("identity", {}).get("name", "Merchant")
    # Split body into chunks for template params
    sentences = body.split(".")
    params = [name]
    for s in sentences[:2]:
        s = s.strip()
        if s and len(s) > 5:
            params.append(s[:80])
    return params[:3]


def _fallback_reply(intent: str, merchant: dict) -> dict:
    """Fallback reply when LLM is unavailable."""
    owner = merchant.get("identity", {}).get("owner_first_name", "")
    name = owner or merchant.get("identity", {}).get("name", "")

    if intent == "action_commit":
        return {
            "action": "send",
            "body": f"Done {name}! Working on it now. Will share the update shortly.",
            "cta": "none",
            "rationale": "Action commitment detected, confirming execution",
        }
    elif intent == "question":
        return {
            "action": "send",
            "body": f"Good question {name}. Let me check and get back to you with the details.",
            "cta": "open_ended",
            "rationale": "Question detected, acknowledged + will follow up",
        }
    elif intent == "auto_reply":
        return {
            "action": "wait",
            "wait_seconds": 1800,
            "rationale": "Auto-reply detected, waiting before retry",
        }
    else:
        return {
            "action": "send",
            "body": "Got it — noted. Anything else I can help with?",
            "cta": "open_ended",
            "rationale": f"General acknowledgment for intent: {intent}",
        }


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BOT_PORT", "8080"))
    host = os.getenv("BOT_HOST", "0.0.0.0")
    logger.info(f"Starting Vera Pro on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
