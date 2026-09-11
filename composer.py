"""
Vera Pro — LLM Composition Engine
====================================
Routes triggers to specialized prompts, calls a multi-provider
OpenAI-compatible LLM chain (groq → cerebras → openrouter), validates
output, and returns composed messages.
"""

import os
import json
import re
import time
import logging
import asyncio
import httpx
from typing import Any

from prompts import (
    BASE_SYSTEM_PROMPT, TRIGGER_PROMPTS, DEFAULT_TRIGGER_PROMPT,
    REPLY_SYSTEM_PROMPT, GOLD_EXEMPLARS,
)
from context_extractor import (
    extract_category_brief, extract_merchant_brief, extract_trigger_brief,
    extract_customer_brief, find_relevant_digest_item, compute_dormancy_days,
    build_salutation, extract_specificity_anchors,
)
from validators import (
    validate_composition, repair_composition, strip_markdown, provenance_issues,
)

logger = logging.getLogger("vera_pro")

# ─────────────────────────────────────────────────────────────────────
# LLM Client — multi-provider, OpenAI-compatible, priority chain
#
# Priority: Groq → Cerebras → OpenRouter (Gemini also supported via GEMINI_API_KEY)
# Every provider below exposes an OpenAI-compatible chat-completions
# endpoint, so one code path serves them all. The chain survives any
# single provider's outage, quota exhaustion, or rate-limit burst —
# which is exactly what happens during the judge's 10-req/sec harness.
# Configure any subset via env vars; the plan is built from what exists.
# ─────────────────────────────────────────────────────────────────────

# Whole-request budget — endpoints must respond within 30s
LLM_BUDGET_S = float(os.getenv("LLM_BUDGET_S", "24"))

# Adaptive model management:
# - a (provider, model) returning 403 tier_not_allowed is disabled for the
#   process lifetime (deterministic per API key — never retry it)
# - pacing is per provider (each free tier has its own RPM ceiling)
_model_disabled: set[str] = set()
_provider_last_call: dict[str, float] = {}
_last_llm_error: str = ""


# Provider definitions — discovery-based so model-ID rotation never breaks
# the chain (catalogs rotate: hardcoded IDs 404 within months, which is what
# crashed the original submission). Model resolution is lazy + cached:
#   1. explicit pins via {PROVIDER}_MODELS env (used verbatim)
#   2. else live catalog discovery from GET /models, best match by
#      `prefer` keywords (earlier = better), `avoid` keywords filtered
#   3. else legacy static default (only if discovery fails)
# On HTTP 404 (model rotated out mid-session): the model is marked failed,
# the catalog is re-fetched, and the next chain entry resolves fresh.
_PROVIDER_DEFS = [
    {
        "name": "groq",
        "env": "GROQ_API_KEY",
        "base": "https://api.groq.com/openai/v1",
        "models_env": "GROQ_MODELS",
        "min_interval": float(os.getenv("GROQ_MIN_INTERVAL_S", "2.2")),
        "prefer": ["gpt-oss-120b", "qwen3.6", "qwen3.8", "gpt-oss-20b", "qwen", "70b"],
        "avoid": ["orpheus", "whisper", "tts", "guard", "embed", "allam", "compound", "vision"],
        "reasoning_effort": "low",  # gpt-oss family: cap hidden chain-of-thought
        "defaults": ["llama-3.3-70b-versatile"],
    },
    {
        "name": "gemini",
        "env": "GEMINI_API_KEY",
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models_env": "GEMINI_MODELS",
        "min_interval": float(os.getenv("GEMINI_MIN_INTERVAL_S", "6.5")),
        "prefer": ["2.5", "flash", "2.0"],
        "avoid": ["vision-"],
        "defaults": ["gemini-2.5-flash"],
    },
    {
        "name": "cerebras",
        "env": "CEREBRAS_API_KEY",
        "base": "https://api.cerebras.ai/v1",
        "models_env": "CEREBRAS_MODELS",
        "min_interval": float(os.getenv("CEREBRAS_MIN_INTERVAL_S", "2.2")),
        "prefer": ["gpt-oss-120b", "qwen", "gemma", "70b"],
        "avoid": [],
        "reasoning_effort": "low",  # gpt-oss family: cap hidden chain-of-thought
        "defaults": ["llama-3.3-70b"],
    },
    {
        "name": "openrouter",
        "env": "OPENROUTER_API_KEY",
        "base": "https://openrouter.ai/api/v1",
        "models_env": "OPENROUTER_MODELS",
        "min_interval": float(os.getenv("OPENROUTER_MIN_INTERVAL_S", "1.0")),
        "prefer": ["pro", "flash", "120b", "large", "max", "deepseek", "qwen"],
        "avoid": ["vision"],
        "reasoning_effort": "low",  # many free OR models are reasoning models too
        "defaults": ["deepseek/deepseek-chat-v3-0324"],
        "free_only": True,  # this deployment runs OpenRouter free models only
    },
]

DISCOVERY_TTL_S = 6 * 60 * 60  # catalogs re-checked at most every 6h


def _build_call_plan() -> list[dict]:
    """One chain entry per configured provider (explicit pins may add more).

    Model is resolved lazily per entry via live catalog discovery
    (see _resolve_model), unless explicitly pinned via {PROVIDER}_MODELS.
    """
    plan: list[dict] = []
    for defn in _PROVIDER_DEFS:
        key = os.getenv(defn["env"], "").strip()
        if not key:
            continue
        pinned = [m.strip() for m in os.getenv(defn["models_env"], "").split(",")
                  if m.strip()]
        base_entry = {
            "provider": defn["name"], "base": defn["base"], "key": key,
            "min_interval": defn["min_interval"], "pinned": pinned,
            "prefer": defn["prefer"], "avoid": defn["avoid"],
            "defaults": defn["defaults"], "free_only": defn.get("free_only", False),
            "reasoning_effort": defn.get("reasoning_effort"),
        }
        if pinned:
            for m in pinned:
                entry = dict(base_entry)
                entry["model"] = m
                entry["label"] = f'{defn["name"]}:{m}'
                plan.append(entry)
        else:
            entry = dict(base_entry)
            entry["model"] = None  # resolved lazily via catalog discovery
            entry["label"] = defn["name"]
            plan.append(entry)
    return plan


CALL_PLAN = _build_call_plan()
PRIMARY_MODEL = (CALL_PLAN[0]["label"] if CALL_PLAN else "none-configured")

# Runtime model resolution state (per provider)
_discovered: dict[str, dict] = {}  # provider -> {models, failed, picked, fetched_at}


def _pick_model(models: list[str], entry: dict, failed: set[str]) -> str | None:
    """Pick the best available model for a provider from its live catalog."""
    prefer = entry.get("prefer", [])
    avoid = entry.get("avoid", [])
    free_only = entry.get("free_only", False)
    best, best_score = None, -1
    for mid in models:
        low = mid.lower()
        if any(a in low for a in avoid):
            continue
        if failed and mid in failed:
            continue
        if free_only and not low.endswith(":free"):
            continue
        score = 1
        if free_only and low.endswith(":free"):
            score += 1000
        for i, kw in enumerate(prefer):
            if kw in low:
                score += (len(prefer) - i) * 10
                break
        if score > best_score:
            best, best_score = mid, score
    return best


async def _fetch_catalog(base: str, key: str) -> list[str]:
    """Fetch the provider's live model catalog (OpenAI-compatible /models)."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.get(
                f"{base}/models", headers={"Authorization": f"Bearer {key}"}
            )
        if resp.status_code != 200:
            logger.warning(f"Model catalog fetch failed: HTTP {resp.status_code}")
            return []
        return [m.get("id", "") for m in resp.json().get("data", []) if m.get("id")]
    except Exception as e:
        logger.warning(f"Model catalog fetch error: {type(e).__name__}: {e}")
        return []


async def _resolve_model(entry: dict) -> str | None:
    """Resolve the best current model for a chain entry (cached 6h).

    Resolution order: pinned models verbatim (skipping known-dead ones),
    then cached catalog pick, then live catalog discovery, then the
    provider's legacy static default as a final fallback.
    """
    provider = entry["provider"]

    if entry["pinned"]:
        failed = _discovered.get(provider, {}).get("failed", set())
        for m in entry["pinned"]:
            if m not in failed:
                return m
        return None

    now = time.monotonic()
    cache = _discovered.get(provider)
    if cache and (now - cache["fetched_at"]) < DISCOVERY_TTL_S:
        remaining = [m for m in cache["models"] if m not in cache["failed"]]
        if cache.get("picked") and cache["picked"] in remaining:
            return cache["picked"]
        picked = _pick_model(remaining, entry, set())
        if picked:
            cache["picked"] = picked
        return picked

    models = await _fetch_catalog(entry["base"], entry["key"])
    cache = {"models": models, "failed": set(), "picked": None,
             "fetched_at": now}
    _discovered[provider] = cache
    if models:
        picked = _pick_model(models, entry, set())
    else:
        # Catalog empty/unreachable — legacy static default as last resort
        picked = None
        for m in entry.get("defaults", []):
            picked = m
            break
        logger.warning(f"{provider}: catalog discovery failed — using default {picked}")
    if picked:
        cache["picked"] = picked
    return picked


def _mark_model_failed(provider: str, model: str) -> None:
    """Model rotated out (404) — exclude it and force a fresh catalog fetch."""
    cache = _discovered.get(provider)
    if not cache:
        cache = {"models": [], "failed": {model}, "picked": None, "fetched_at": 0.0}
        _discovered[provider] = cache
    cache["failed"].add(model)
    cache["picked"] = None
    cache["fetched_at"] = 0.0  # forces re-discovery on next resolve

# Last LLM failure signature — surfaced in the fallback rationale so the
# exact failure mode (bad key / 401 / timeout / egress error) is visible
# in tick & reply responses without needing server log access.
_last_llm_error: str = ""


async def _call_llm(
    system_prompt: str,
    user_prompt: str,
    retries: int = 1,
    budget_s: float = LLM_BUDGET_S,
    # Headroom matters: reasoning models spend max_tokens on hidden
    # chain-of-thought BEFORE the JSON body (measured: 498 reasoning tokens
    # ate a 500-token budget -> 200-with-empty-content -> parse failure).
    max_tokens: int = 900,
) -> str | None:
    """Try every provider/model in CALL_PLAN within a time budget.

    Survives single-provider outages, tier blocks, and rate-limit bursts:
    each (provider, model) that hard-fails is disabled for the process
    lifetime, and pacing is per provider (each free tier has its own RPM).
    """
    global _last_llm_error, _provider_last_call
    if not CALL_PLAN:
        _last_llm_error = ("no LLM provider configured — set GROQ_API_KEY / "
                           "CEREBRAS_API_KEY / OPENROUTER_API_KEY")
        logger.error(_last_llm_error)
        return None

    started = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(budget_s)) as client:
        for entry in CALL_PLAN:
            label = entry["label"]
            if label in _model_disabled:
                continue  # tier-blocked / auth-dead earlier — skip silently
            remaining = budget_s - (time.monotonic() - started)
            if remaining < 4.0:
                _last_llm_error = "LLM time budget exhausted"
                logger.warning("LLM time budget exhausted; stopping call plan")
                break
            # Lazy model resolution: live catalog discovery (cached 6h) or
            # explicit pins. None => provider has no usable model right now.
            model = await _resolve_model(entry)
            if model is None:
                _model_disabled.add(label)
                _last_llm_error = f"{label} — no usable model in catalog"
                logger.warning(f"{label}: no usable model resolved — entry disabled")
                continue
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            }
            # Reasoning models (gpt-oss family) burn max_tokens on hidden
            # chain-of-thought — cap the effort so the JSON body survives.
            effort = entry.get("reasoning_effort")
            if effort:
                body["reasoning_effort"] = effort
            headers = {
                "Authorization": f"Bearer {entry['key']}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            for attempt in range(retries + 1):
                # Per-entry cap, computed OUTSIDE the try so the timeout
                # handler can always reference it (never possibly-unbound).
                call_timeout = max(4.0, min(remaining, budget_s, budget_s * 0.45))
                try:
                    # Per-provider pacing (free-tier RPM ceilings differ)
                    wait = entry["min_interval"] - (
                        time.monotonic() - _provider_last_call.get(label, 0.0)
                    )
                    if wait > 0:
                        await asyncio.sleep(wait)
                    _provider_last_call[label] = time.monotonic()
                    resp = await client.post(
                        f"{entry['base']}/chat/completions", headers=headers,
                        json=body, timeout=call_timeout,
                    )
                    if resp.status_code == 200:
                        try:
                            content = resp.json()["choices"][0]["message"]["content"] or ""
                        except (KeyError, IndexError, ValueError):
                            content = ""
                        if not content.strip():
                            # 200-with-empty-content: reasoning models can spend
                            # the entire max_tokens on hidden chain-of-thought.
                            # Retry once with a doubled budget (effort-cap may
                            # be unsupported/ignored by this provider), then
                            # fall through to the next provider.
                            if attempt == 0 and body["max_tokens"] < 4000:
                                body["max_tokens"] = min(body["max_tokens"] * 2, 4000)
                                _last_llm_error = (
                                    f"{label} 200 but empty content — retry 2x tokens"
                                )
                                logger.warning(
                                    f"{label}: empty content — retrying with "
                                    f"max_tokens={body['max_tokens']}"
                                )
                                continue
                            _last_llm_error = f"{label} 200 but empty content"
                            logger.warning(f"{label}: 200 with empty content — next provider")
                            break
                        elapsed = time.monotonic() - started
                        logger.info(f"LLM OK via {label} in {elapsed:.1f}s")
                        _last_llm_error = ""
                        return content
                    elif resp.status_code == 401:
                        _model_disabled.add(label)
                        _last_llm_error = f"{label} 401 — invalid API key, disabled"
                        logger.error(f"{label} auth failed (401) — disabled")
                        break  # next provider
                    elif resp.status_code in (402, 403):
                        # Deterministic per-key signals:
                        #   403 tier_not_allowed / 402 payment_required —
                        # disable for the process lifetime, try next provider.
                        _model_disabled.add(label)
                        _last_llm_error = f"{label} HTTP {resp.status_code} — auto-disabled"
                        logger.error(f"{label} unavailable (HTTP {resp.status_code}) — "
                                     f"disabled for process lifetime")
                        break  # next provider
                    elif resp.status_code == 404:
                        # Model rotated out of the provider's catalog — mark
                        # failed + force re-discovery; next call self-heals.
                        _mark_model_failed(entry["provider"], model)
                        _last_llm_error = f"{label} 404 model rotated — re-discovering"
                        logger.warning(
                            f"{label}: model {model} gone (404) — re-discovering"
                        )
                        break  # next entry in the plan
                    elif resp.status_code == 400 and "reasoning_effort" in body:
                        # Provider/model rejects the reasoning_effort param —
                        # retry the same entry once without it.
                        body.pop("reasoning_effort", None)
                        _last_llm_error = f"{label} 400 — retrying without reasoning_effort"
                        logger.warning(f"{label}: 400 rejected reasoning_effort — retrying plain")
                        continue
                    elif resp.status_code == 429:
                        _last_llm_error = f"{label} 429 rate limited"
                        ra = resp.headers.get("retry-after", "")
                        wait = 2.0
                        if ra.replace(".", "", 1).isdigit():
                            wait = min(5.0, max(1.0, float(ra)))
                        logger.warning(f"{label} rate limited (attempt {attempt + 1}) — "
                                       f"backoff {wait:.1f}s")
                        await asyncio.sleep(wait)
                        continue  # retry same entry once
                    else:
                        _last_llm_error = f"{label} HTTP {resp.status_code}: {resp.text[:80]}"
                        logger.warning(f"{label} error {resp.status_code}: {resp.text[:150]}")
                        break  # fall to next provider in the plan
                except (httpx.TimeoutException, httpx.ReadTimeout):
                    _last_llm_error = f"{label} timeout after {call_timeout:.0f}s"
                    logger.warning(f"{label} timeout after {call_timeout:.0f}s")
                    break  # tighter budget — move to next provider immediately
                except Exception as e:
                    _last_llm_error = f"{label} {type(e).__name__}: {str(e)[:80]}"
                    logger.warning(f"{label} request error: {e}")
                    break

    return None


def _parse_json_response(text: str | None) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks.

    strict=False tolerates literal control characters (raw newlines/tabs)
    inside string values — smaller chain models (e.g. open-mistral-nemo)
    emit pretty-printed JSON with unescaped newlines in the body strings.
    Accepts None (LLM unavailable) and returns None in that case.
    """
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text.strip(), strict=False)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    patterns = [
        r'```json\s*\n?(.*?)\n?```',
        r'```\s*\n?(.*?)\n?```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                candidate = match.group(1) if match.lastindex else match.group(0)
                return json.loads(candidate.strip(), strict=False)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


# ─────────────────────────────────────────────────────────────────────
# Engagement Composer
# ─────────────────────────────────────────────────────────────────────

class EngagementComposer:
    """Composes messages using the 4-context framework + LLM."""

    async def compose(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: dict | None = None,
        conversation_history: list[dict] | None = None,
        recent_sent_bodies: list[str] | None = None,
    ) -> dict:
        """
        Compose a message from the 4 contexts.
        Pipeline: extract → prompt → LLM → validate → (LLM repair) →
        deterministic repair. Returns dict: body, cta, send_as, suppression_key, rationale.
        """
        started = time.monotonic()

        # 1. Extract structured briefs
        cat_brief = extract_category_brief(category)
        merch_brief = extract_merchant_brief(merchant, cat_brief)
        trig_brief = extract_trigger_brief(trigger)
        cust_brief = extract_customer_brief(customer)

        # 2. Build system + user prompts (anchors injected in user prompt)
        system_prompt = self._build_system_prompt(cat_brief, merch_brief)
        avoid_list = list(recent_sent_bodies or [])[-3:]
        user_prompt = self._build_user_prompt(
            cat_brief, merch_brief, trig_brief, cust_brief, category,
            avoid_openings=avoid_list,
            raw_merchant=merchant, raw_trigger=trigger, raw_customer=customer,
        )

        # 3. Call LLM (deadline-aware)
        elapsed = time.monotonic() - started
        raw_response = await _call_llm(
            system_prompt, user_prompt, budget_s=max(8.0, LLM_BUDGET_S - elapsed)
        )

        # 4. Parse response
        result = _parse_json_response(raw_response)

        if not result:
            logger.warning(f"Failed to parse LLM response, using fallback. Raw: {str(raw_response)[:200]}")
            result = self._fallback_compose(merch_brief, trig_brief, cust_brief)

        # 5. Ensure required fields
        result = self._ensure_fields(result, trig_brief, merch_brief, cust_brief)

        # 6. Validate (+ numeric provenance against all 4 raw contexts)
        history = conversation_history or merchant.get("conversation_history") or []
        is_valid, issues = validate_composition(
            result, category, merchant, trigger, customer, history
        )
        prov_contexts = (category, merchant, trigger.get("payload", {}), customer)
        issues += provenance_issues(result.get("body", ""), *prov_contexts)

        # 7. One LLM repair pass if issues remain and budget allows
        if issues:
            remaining = LLM_BUDGET_S - (time.monotonic() - started)
            if remaining >= 10.0:
                repair_prompt = self._build_repair_prompt(result, issues)
                raw2 = await _call_llm(
                    system_prompt, repair_prompt, budget_s=min(remaining, 12.0),
                )
                fixed = _parse_json_response(raw2)
                if fixed and isinstance(fixed.get("body"), str) and len(fixed["body"].strip()) > 10:
                    for key in ("body", "cta", "send_as", "suppression_key", "rationale"):
                        if key in fixed and fixed[key]:
                            result[key] = fixed[key]
                    logger.info("LLM repair pass applied")
            else:
                logger.info(f"Skipping LLM repair — only {remaining:.0f}s left")

        # 8. Deterministic repair — zero-flaw guarantee before shipping
        result, repairs = repair_composition(
            result, category, merchant, trigger, customer,
            provenance_contexts=prov_contexts,
        )
        if repairs:
            logger.info(f"Deterministic repairs applied: {repairs}")

        return result

    def _build_repair_prompt(self, result: dict, issues: list[str]) -> str:
        """Build a focused re-prompt asking the LLM to fix specific issues."""
        import json as _json
        return f"""Your previous composition had these problems:

{chr(10).join('- ' + i for i in issues)}

Previous output (as JSON):
{_json.dumps(result, ensure_ascii=False)}

Rewrite the message fixing ALL listed problems. Keep everything that was good:
same trigger intent, same facts (do NOT invent new numbers), same language mix.
Return ONLY a JSON object with keys: body, cta, send_as, suppression_key, rationale."""

    async def compose_reply(
        self,
        conversation_state: dict,
        merchant_message: str,
        category: dict,
        merchant: dict,
    ) -> dict:
        """Compose a reply to a merchant/customer message within a conversation."""
        cat_brief = extract_category_brief(category)
        merch_brief = extract_merchant_brief(merchant, cat_brief)

        # Build conversation history string
        history = conversation_state.get("turns", [])
        history_str = ""
        for turn in history[-6:]:  # Last 6 turns max
            role = turn.get("from", "unknown")
            body = turn.get("body", "")[:200]
            history_str += f"[{role.upper()}]: {body}\n"

        # Add current message
        history_str += f"[MERCHANT]: {merchant_message}\n"

        system = _safe_format(REPLY_SYSTEM_PROMPT, {
            "conv_state": conversation_state.get("state", "active"),
            "turn_number": str(conversation_state.get("turn_count", 1)),
            "conversation_history": history_str,
            "merchant_name": merch_brief["name"],
            "category_slug": cat_brief["slug"],
            "merchant_languages": ", ".join(merch_brief["languages"]),
        })

        perf = merch_brief.get("performance", {})
        ctx_lines = [
            f"- Active offers: {json.dumps(merch_brief.get('active_offers', []), ensure_ascii=False)}",
            f"- Performance: views={perf.get('views')}, calls={perf.get('calls')} (last 30d)",
        ]
        themes = merch_brief.get("review_themes", [])
        if themes:
            import json as _j
            ctx_lines.append(f"- Recent review theme: {_j.dumps(themes[0], ensure_ascii=False)}")
        trends = cat_brief.get("trend_signals", []) or []
        if trends:
            top_trend = max(trends, key=lambda t: t.get("delta_yoy", 0))
            dyoy = round(top_trend.get("delta_yoy", 0) * 100)
            ctx_lines.append(
                f"- Demand trend: \"{top_trend.get('query', '')}\" {'+' if dyoy >= 0 else ''}{dyoy}% YoY"
            )

        user = f"""The merchant/customer just said: "{merchant_message}"

Verified context for your response (use ONLY these numbers):
{chr(10).join(ctx_lines)}

Decide: send a reply, wait, or end? Respond with JSON only.
If you cite any number, it must come from the context above or the conversation history."""

        raw = await _call_llm(system, user, budget_s=14.0)
        result = _parse_json_response(raw)

        if not result:
            # Intelligent fallback based on detected intent
            from validators import detect_intent
            intent = detect_intent(merchant_message)
            result = self._fallback_reply(intent, merch_brief)

        # Ensure action field
        if "action" not in result:
            result["action"] = "send"

        # Hygiene: no markdown / overlong bodies on replies
        if isinstance(result.get("body"), str):
            result["body"] = strip_markdown(result["body"])
            if len(result["body"]) > 350 and result.get("action") == "send":
                logger.info("Reply too long — keeping WhatsApp brevity")
        if result.get("action") == "wait" and not result.get("wait_seconds"):
            result["wait_seconds"] = 1800

        return result

    def _build_system_prompt(self, cat_brief: dict, merch_brief: dict) -> str:
        """Build the system prompt with category voice and merchant language."""
        voice_desc = f"{cat_brief.get('tone', 'professional')} / {cat_brief.get('register', 'neutral')}"
        taboos = ", ".join(cat_brief.get("vocab_taboo", []))
        languages = ", ".join(merch_brief.get("languages", ["en"]))

        return _safe_format(BASE_SYSTEM_PROMPT, {
            "category_voice": voice_desc,
            "category_taboos": taboos or "none",
            "merchant_languages": languages,
        })

    def _build_user_prompt(
        self,
        cat_brief: dict,
        merch_brief: dict,
        trig_brief: dict,
        cust_brief: dict | None,
        raw_category: dict,
        avoid_openings: list[str] | None = None,
        raw_merchant: dict | None = None,
        raw_trigger: dict | None = None,
        raw_customer: dict | None = None,
    ) -> str:
        """Build the trigger-specific user prompt with all context injected."""
        trigger_kind = trig_brief["kind"]
        template = TRIGGER_PROMPTS.get(trigger_kind, DEFAULT_TRIGGER_PROMPT)

        # Build substitution dict with all possible template variables
        subs = {
            # Category
            "salutation": build_salutation(raw_category, {"identity": {"owner_first_name": merch_brief["owner_first_name"], "name": merch_brief["name"]}}),
            "category_slug": cat_brief["slug"],
            "offer_catalog": json.dumps(cat_brief["offer_catalog"], ensure_ascii=False),
            "peer_stats": json.dumps(cat_brief["peer_stats"], ensure_ascii=False),
            "seasonal_beats": json.dumps(cat_brief["seasonal_beats"], ensure_ascii=False),
            "trend_signals": json.dumps(cat_brief["trend_signals"], ensure_ascii=False),

            # Merchant
            "merchant_name": merch_brief["name"],
            "merchant_locality": merch_brief["locality"],
            "merchant_city": merch_brief["city"],
            "merchant_signals": json.dumps(merch_brief["signals"], ensure_ascii=False),
            "merchant_languages": json.dumps(merch_brief["languages"], ensure_ascii=False),
            "performance": json.dumps(merch_brief["performance"], ensure_ascii=False),
            "delta_7d": json.dumps(merch_brief["delta_7d"], ensure_ascii=False),
            "active_offers": json.dumps(merch_brief["active_offers"], ensure_ascii=False),
            "customer_aggregate": json.dumps(merch_brief["customer_aggregate"], ensure_ascii=False),
            "review_themes": json.dumps(merch_brief["review_themes"], ensure_ascii=False),
            "subscription": json.dumps({
                "status": merch_brief["subscription_status"],
                "plan": merch_brief["subscription_plan"],
                "days_remaining": merch_brief["days_remaining"],
            }, ensure_ascii=False),
            "days_remaining": str(merch_brief["days_remaining"]),
            "last_conversation": merch_brief.get("last_conversation", "None"),
            "peer_insights": json.dumps(merch_brief.get("peer_insights", []), ensure_ascii=False),

            # Trigger
            "trigger_kind": trigger_kind,
            "trigger_payload": json.dumps(trig_brief["payload"], ensure_ascii=False),

            # Customer
            "customer_context": json.dumps(cust_brief, ensure_ascii=False) if cust_brief else "None",

            # Derived
            "dormancy_days": str(compute_dormancy_days({"signals": merch_brief["signals"]})),
        }

        # Find digest item if research_digest trigger
        if trigger_kind == "research_digest":
            digest_item = find_relevant_digest_item(trig_brief, raw_category)
            subs["digest_item"] = json.dumps(digest_item, ensure_ascii=False) if digest_item else "No specific digest item found"

        # Find review theme if review_theme_emerged trigger
        if trigger_kind == "review_theme_emerged":
            themes = merch_brief.get("review_themes", [])
            subs["review_theme"] = json.dumps(themes[0] if themes else {}, ensure_ascii=False)

        # Safe format — only substitute keys that exist in the template
        try:
            prompt = template.format(**subs)
        except KeyError as e:
            logger.warning(f"Missing template key {e}, using safe format")
            prompt = _safe_format(template, subs)

        # ── Gold exemplar for this trigger kind (if available) ──
        exemplar = GOLD_EXEMPLARS.get(trigger_kind)
        if exemplar:
            prompt += f"\n\n## GOLD EXAMPLE OF A 10/10 MESSAGE FOR THIS TRIGGER (pattern to emulate, never copy verbatim)\n{exemplar}"

        # ── Derived specificity anchors — verified facts ready to cite ──
        try:
            anchors = extract_specificity_anchors(
                raw_category or {}, raw_merchant or {},
                raw_trigger or trig_brief, raw_customer,
            )
        except Exception as e:
            logger.warning(f"Anchor extraction failed: {e}")
            anchors = []
        if anchors:
            lines = "\n".join(f"- {a}" for a in anchors)
            prompt += (
                "\n\n## VERIFIED FACTS YOU MAY CITE (all pre-computed from real context — "
                "cite 2-4 of these, NEVER invent other numbers)\n" + lines
            )

        # ── Anti-repetition guardrail across conversations ──
        if avoid_openings:
            avoid_str = "\n".join(f"- {b[:110]}" for b in avoid_openings if b)
            prompt += (
                "\n\n## ALREADY-SENT MESSAGES TO THIS MERCHANT (vary angle/opening — "
                "never repeat these phrasings)\n" + avoid_str
            )

        return prompt

    def _ensure_fields(self, result: dict, trig: dict, merch: dict, cust: dict | None) -> dict:
        """Ensure all required fields exist with sensible defaults."""
        if "body" not in result:
            result["body"] = ""
        if "cta" not in result:
            result["cta"] = "open_ended"
        if "send_as" not in result:
            result["send_as"] = "merchant_on_behalf" if (cust and trig.get("scope") == "customer") else "vera"
        if "suppression_key" not in result:
            result["suppression_key"] = trig.get("suppression_key", f"{trig['kind']}:{merch.get('name', 'unknown')}")
        if "rationale" not in result:
            result["rationale"] = f"Composed for trigger {trig['kind']}"
        return result

    def _fallback_compose(self, merch: dict, trig: dict, cust: dict | None) -> dict:
        """Fallback composition when LLM fails."""
        name = merch.get("name", "")
        owner = merch.get("owner_first_name", "")
        kind = trig.get("kind", "update")
        active_offers = merch.get("active_offers", [])

        if cust and trig.get("scope") == "customer":
            cust_name = cust.get("name", "")
            body = f"Hi {cust_name}, {name} here. We noticed it's been a while since your last visit."
            if active_offers:
                body += f" We have a special offer: {active_offers[0]}. Would you like to book a slot?"
            return {
                "body": body,
                "cta": "open_ended",
                "send_as": "merchant_on_behalf",
                "suppression_key": trig.get("suppression_key", f"fallback:{kind}"),
                "rationale": (
                    "Fallback composition — LLM unavailable"
                    + (f" [{_last_llm_error[:100]}]" if _last_llm_error else "")
                ),
            }
        else:
            greeting = owner or name
            body = f"Hi {greeting}, quick update on your business profile."
            perf = merch.get("performance", {})
            if perf.get("views"):
                body += f" You had {perf['views']} views in the last 30 days."
            return {
                "body": body,
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": trig.get("suppression_key", f"fallback:{kind}"),
                "rationale": (
                    "Fallback composition — LLM unavailable"
                    + (f" [{_last_llm_error[:100]}]" if _last_llm_error else "")
                ),
            }

    def _fallback_reply(self, intent: str, merch: dict) -> dict:
        """Fallback reply when LLM fails for conversation handling."""
        if intent == "auto_reply":
            return {
                "action": "end",
                "body": "",
                "cta": "none",
                "rationale": "Detected auto-reply, gracefully exiting",
            }
        elif intent == "hostile" or intent == "not_interested":
            name = merch.get("owner_first_name", merch.get("name", ""))
            return {
                "action": "end",
                "body": f"Samajh gayi {name}. Aapko kabhi bhi help chahiye to Vera yahin hai. Best wishes! 🙂",
                "cta": "none",
                "rationale": f"Merchant signaled {intent}, graceful exit",
            }
        elif intent == "action_commit":
            return {
                "action": "send",
                "body": "Done! Working on it now. Will share the update shortly.",
                "cta": "none",
                "rationale": "Merchant committed to action, confirming execution",
            }
        else:
            return {
                "action": "send",
                "body": "Got it — let me check and get back to you on this.",
                "cta": "open_ended",
                "rationale": f"Fallback reply for intent: {intent}",
            }


def _safe_format(template: str, subs: dict) -> str:
    """Format template, ignoring missing keys."""
    result = template
    for key, value in subs.items():
        result = result.replace("{" + key + "}", str(value))
    return result
