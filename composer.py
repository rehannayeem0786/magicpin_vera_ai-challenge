"""
Vera Pro — LLM Composition Engine
====================================
Routes triggers to specialized prompts, calls Mistral API,
validates output, and returns composed messages.
"""

import os
import json
import re
import time
import logging
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
# LLM Client — Mistral API
# ─────────────────────────────────────────────────────────────────────

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
_env_model = os.getenv("MISTRAL_MODEL", "").strip()
# Chain order: medium is primary — measured ~0.9-3s vs large's instability
# (2026-08: mistral-large-latest hung 60s+ platform-side, see probe logs).
# medium → large → nemo keeps frontier quality in the chain while
# guaranteeing a sub-10s composition in the healthy path.
MODEL_CHAIN = [m for m in [
    _env_model or "mistral-medium-latest",
    "mistral-large-latest",
    "open-mistral-nemo",
] if m]
PRIMARY_MODEL = MODEL_CHAIN[0]
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# Whole-request budget — endpoints must respond within 30s
LLM_BUDGET_S = float(os.getenv("LLM_BUDGET_S", "24"))

# Last LLM failure signature — surfaced in the fallback rationale so the
# exact failure mode (bad key / 401 / timeout / egress error) is visible
# in tick & reply responses without needing server log access.
_last_llm_error: str = ""


async def _call_llm(
    system_prompt: str,
    user_prompt: str,
    retries: int = 1,
    budget_s: float = LLM_BUDGET_S,
    max_tokens: int = 500,
) -> str | None:
    """Call Mistral API trying each model in MODEL_CHAIN within a time budget."""
    global _last_llm_error
    if not MISTRAL_API_KEY:
        _last_llm_error = "no MISTRAL_API_KEY configured"
        logger.error("No MISTRAL_API_KEY configured!")
        return None

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=httpx.Timeout(budget_s)) as client:
        for model in MODEL_CHAIN:
            remaining = budget_s - (time.monotonic() - started)
            if remaining < 4.0:
                _last_llm_error = "LLM time budget exhausted"
                logger.warning("LLM time budget exhausted; stopping model chain")
                break
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            }
            for attempt in range(retries + 1):
                try:
                    # Per-model cap: no single hanging model may consume the
                    # whole budget — the next chain model must always get a turn.
                    call_timeout = max(4.0, min(remaining, budget_s, budget_s * 0.45))
                    resp = await client.post(
                        MISTRAL_URL, headers=headers, json=body,
                        timeout=call_timeout,
                    )
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        elapsed = time.monotonic() - started
                        logger.info(f"LLM OK via {model} in {elapsed:.1f}s")
                        _last_llm_error = ""
                        return content
                    elif resp.status_code == 401:
                        _last_llm_error = "401 auth failed — invalid API key"
                        logger.error("Mistral auth failed (401) — check API key")
                        return None
                    elif resp.status_code == 429:
                        _last_llm_error = f"{model} 429 rate limited"
                        logger.warning(f"{model} rate limited")
                        import asyncio
                        await asyncio.sleep(0.8)
                        continue  # retry same model once
                    else:
                        _last_llm_error = f"{model} HTTP {resp.status_code}: {resp.text[:80]}"
                        logger.warning(f"{model} error {resp.status_code}: {resp.text[:150]}")
                        break  # fall to next model in chain
                except (httpx.TimeoutException, httpx.ReadTimeout):
                    _last_llm_error = f"{model} timeout after {call_timeout:.0f}s"
                    logger.warning(f"{model} timeout after {call_timeout:.0f}s")
                    break  # tighter budget — move to next model immediately
                except Exception as e:
                    _last_llm_error = f"{model} {type(e).__name__}: {str(e)[:80]}"
                    logger.warning(f"{model} request error: {e}")
                    break

    return None


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text.strip())
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
                return json.loads(candidate.strip())
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
