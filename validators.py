"""
Vera Pro — Post-Composition Validators
========================================
Validates LLM output for quality, compliance, and anti-hallucination.
"""

import re
import json as _json
from typing import Any


def validate_composition(
    result: dict,
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate a composed message against quality rules.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    body = result.get("body", "")

    # 1. Empty body check
    if not body or len(body.strip()) < 10:
        issues.append("CRITICAL: Message body is empty or too short")

    # 2. Voice taboo check
    voice = category.get("voice", {})
    taboos = voice.get("vocab_taboo", [])
    body_lower = body.lower()
    for taboo in taboos:
        taboo_lower = taboo.lower().split("(")[0].strip()  # Remove parenthetical notes
        if taboo_lower and taboo_lower in body_lower:
            issues.append(f"TABOO VIOLATION: Message contains '{taboo_lower}' (forbidden for {category.get('slug', 'this category')})")

    # 3. CTA shape check
    cta = result.get("cta", "")
    if cta not in ("open_ended", "binary_yes_stop", "none", ""):
        issues.append(f"INVALID CTA: '{cta}' — must be open_ended, binary_yes_stop, or none")

    # 4. send_as check
    send_as = result.get("send_as", "")
    trigger_scope = trigger.get("scope", "merchant")
    if trigger_scope == "customer" and customer:
        if send_as != "merchant_on_behalf":
            issues.append("SEND_AS MISMATCH: Customer-facing trigger should use send_as='merchant_on_behalf'")
    elif not customer and send_as == "merchant_on_behalf":
        issues.append("SEND_AS MISMATCH: No customer context but send_as='merchant_on_behalf'")

    # 5. Anti-repetition check
    if conversation_history:
        for prev in conversation_history:
            prev_body = prev.get("body", "")
            if prev_body and body.strip() == prev_body.strip():
                issues.append("REPETITION: Message is identical to a previous message in conversation")
                break

    # 6. Multiple CTA check
    cta_patterns = [
        r"reply\s+\d+\s+for.*reply\s+\d+\s+for.*reply\s+\d+\s+for",
        r"option\s*[a-d].*option\s*[a-d].*option\s*[a-d]",
    ]
    for pattern in cta_patterns:
        if re.search(pattern, body_lower):
            issues.append("MULTIPLE CTAS: Message appears to have 3+ options (limit to binary or open-ended)")

    # 7. Promotional tone check
    promo_patterns = [
        r"amazing\s*(deal|offer|discount)",
        r"don'?t\s+miss\s+(out|this)",
        r"hurry",
        r"limited\s+time\s+offer",
        r"act\s+now",
        r"best\s+deal\s+ever",
        r"unbelievable",
        r"once\s+in\s+a\s+lifetime",
    ]
    for pattern in promo_patterns:
        if re.search(pattern, body_lower):
            issues.append(f"PROMOTIONAL TONE: Detected promotional language matching '{pattern}'")

    # 7b. Markdown leakage check (WhatsApp renders *single* asterisks as bold;
    #     double asterisks / backticks show literally — sloppy on WhatsApp)
    md_patterns = [r"\*\*", r"`{1,3}", r"^#{1,3}\s"]
    for pattern in md_patterns:
        if re.search(pattern, body):
            issues.append(f"MARKDOWN LEAK: Message contains markdown artifact matching '{pattern}'")

    # 7c. Message length check — wall-of-text penalty risk
    if len(body) > 480:
        issues.append("TOO LONG: Message exceeds 480 chars — trim toward WhatsApp brevity (~60 words)")

    # 8. Long preamble check
    preamble_patterns = [
        r"^(hi|hello|hey).*i\s+hope\s+(you'?re|you\s+are)\s+doing\s+well",
        r"^i'?m\s+reaching\s+out\s+today\s+to",
        r"^i\s+wanted\s+to\s+take\s+a\s+moment\s+to",
    ]
    for pattern in preamble_patterns:
        if re.search(pattern, body_lower):
            issues.append("LONG PREAMBLE: Message starts with generic greeting (keep it direct)")

    # 9. Suppression key check
    if not result.get("suppression_key"):
        issues.append("MISSING SUPPRESSION KEY: Required for dedup")

    is_valid = not any(issue.startswith("CRITICAL") for issue in issues)
    return is_valid, issues


def detect_auto_reply(message: str, conversation_history: list[dict] | None = None) -> bool:
    """
    Detect if a merchant's reply is an auto-reply.
    Heuristics:
    1. Matches common auto-reply patterns
    2. Same message repeated in conversation history
    """
    msg_lower = message.lower().strip()

    # Common auto-reply patterns
    auto_patterns = [
        r"thank\s+(you|u)\s+for\s+(contacting|reaching|messaging|writing)",
        r"our\s+team\s+will\s+(respond|get\s+back|reply)",
        r"we\s+(have\s+)?received\s+your\s+(message|query|request)",
        r"automated\s+(response|reply|message|assistant)",
        r"we'?ll\s+get\s+back\s+to\s+you",
        r"your\s+(message|call|query)\s+is\s+important",
        r"currently\s+(unavailable|away|busy|closed)",
        r"business\s+hours\s+are",
        r"leave\s+(a\s+message|your\s+details)",
        r"aapki\s+jaankari\s+ke\s+liye.*shukriya",  # Hindi auto-reply
        r"main\s+ek\s+automated\s+assistant\s+hoon",
    ]

    for pattern in auto_patterns:
        if re.search(pattern, msg_lower):
            return True

    # Check for verbatim repetition in history
    if conversation_history:
        repeat_count = 0
        for prev in conversation_history:
            if prev.get("from") in ("merchant", "customer"):
                prev_msg = prev.get("body", "").lower().strip()
                # Fuzzy match — if 80%+ similar
                if prev_msg and (prev_msg == msg_lower or
                                 _similarity(prev_msg, msg_lower) > 0.8):
                    repeat_count += 1
        if repeat_count >= 2:
            return True
        # NOTE: verbatim-repetition policy lives in conversation_handlers
        # (canonical normalization + immediate-exit rule) — do not duplicate
        # here; a naive x1 shortcut false-positives on the caller's own turn.

    return False


def detect_intent(message: str) -> str:
    """
    Detect merchant's intent from their reply.
    Returns one of: action_commit, question, not_interested, hostile, auto_reply, engaged, unclear
    """
    msg_lower = message.lower().strip()

    # Auto-reply check
    if detect_auto_reply(message):
        return "auto_reply"

    # Hostile detection
    hostile_patterns = [
        r"stop\s+(messaging|texting|bothering|spamming)",
        r"don'?t\s+(contact|message|text|call)\s+me",
        r"useless\s+(spam|service|bot)",
        r"f+\s*(off|u+ck)",
        r"block",
        r"report",
        r"harassment",
    ]
    for pattern in hostile_patterns:
        if re.search(pattern, msg_lower):
            return "hostile"

    # Not interested detection
    not_interested_patterns = [
        r"not\s+interested",
        r"no\s+thanks?",
        r"nahi\s+chahiye",  # Hindi: don't want
        r"nhi\s+chahiye",
        r"band\s+karo",  # Hindi: stop
        r"mat\s+bhejo",  # Hindi: don't send
        r"don'?t\s+need",
        r"remove\s+me",
        r"unsubscribe",
    ]
    for pattern in not_interested_patterns:
        if re.search(pattern, msg_lower):
            return "not_interested"

    # Action commitment detection
    action_patterns = [
        r"^(yes|yeah|yep|ya|haan|ha|ji|sure|ok|okay|done|go\s+ahead|proceed|let'?s\s+do\s+it|karo|kar\s+do|chal|theek|thik)\b",
        r"(let'?s|lets)\s+(do|start|go|proceed|begin)",
        r"go\s+ahead",
        r"(kar\s+do|kardo|kar\s+dena|kijiye)",  # Hindi: do it
        r"(haan|ha)\s*(please|bhai|sir|ma'?am)?$",
        r"what'?s\s+next",
        r"how\s+do\s+i\s+(start|begin|proceed|join)",
        r"sign\s+me\s+up",
        r"i'?m\s+in",
        r"(mujhe|muje)\s+(join|start|add)\s+karna\s+hai",
        # English explicit intent-to-act (the brief's #1 handoff failure)
        r"(want to|i'?d like to|would like to|wish to)\s+(join|start|begin|proceed|try|signup|sign\s?up)",
        r"(i'?m|im|we'?re|we are)\s+(interested|ready|excited)",
        r"(count me in|add me|register me|onboard me|book\s+(a\s+)?(slot|demo|call))",
    ]
    for pattern in action_patterns:
        if re.search(pattern, msg_lower):
            return "action_commit"

    # Question detection
    question_patterns = [
        r"\?$",
        r"^(what|how|when|where|why|which|who|can|could|would|is|are|do|does|will|kya|kaise|kab|kahan|kitna|kitne|kaun)\b",
    ]
    for pattern in question_patterns:
        if re.search(pattern, msg_lower):
            return "question"

    # Engaged (positive response that's not a clear action commit)
    engaged_patterns = [
        r"(interesting|good|nice|great|accha|achha|bahut\s+accha|helpful|useful)",
        r"tell\s+me\s+more",
        r"(batao|bataiye|batana)",  # Hindi: tell me
        r"(send|share|show)\s+(me|it|this)",
    ]
    for pattern in engaged_patterns:
        if re.search(pattern, msg_lower):
            return "engaged"

    return "unclear"


def _similarity(a: str, b: str) -> float:
    """Simple character-level similarity ratio."""
    if not a or not b:
        return 0.0
    # Use set intersection as a fast approximation
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


# ─────────────────────────────────────────────────────────────────────
# Deterministic repair — last-resort safety net so NO bad message ships
# ─────────────────────────────────────────────────────────────────────

_MARKDOWN_CLEANUP = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),   # **bold**
    (re.compile(r"\*(.+?)\*"), r"\1"),       # *bold*
    (re.compile(r"__(.+?)__"), r"\1"),       # __bold__
    (re.compile(r"_(.+?)_"), r"\1"),         # _italic_
    (re.compile(r"`{1,3}([^`]*)`{1,3}"), r"\1"),  # code spans
]


def strip_markdown(text: str) -> str:
    """Remove markdown artifacts that render poorly on WhatsApp."""
    for pattern, repl in _MARKDOWN_CLEANUP:
        text = pattern.sub(repl, text)
    text = re.sub(r"^#{1,3}\s", "", text.strip())
    text = re.sub(r"[ \t]{2,}", " ", text)          # collapse doubles
    text = re.sub(r"\s+([,.!?])", r"\1", text)      # space before punctuation
    return text.strip()


def remove_taboos(body: str, taboos: list[str]) -> tuple[str, list[str]]:
    """Remove forbidden phrases. Returns (cleaned_body, removed_phrases)."""
    removed = []
    cleaned = body
    for taboo in taboos:
        t = taboo.lower().split("(")[0].strip()
        if t and t in cleaned.lower():
            # Remove the whole sentence containing the taboo (safer than mid-sentence surgery)
            sentences = re.split(r"(?<=[.!?\n])\s+", cleaned)
            kept = []
            for s in sentences:
                if t in s.lower():
                    removed.append(t)
                    continue
                kept.append(s)
            cleaned = " ".join(k for k in kept if k.strip())
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, removed


def enforce_cta_last(result: dict, trigger_kind: str = "") -> str:
    """Ensure CTA shape matches intent; append/repair binary CTA if missing.
    Returns description of any fix applied ('' if none)."""
    body = result.get("body", "")
    cta = result.get("cta", "")
    lower = body.lower()
    fix = ""

    if cta == "binary_yes_stop":
        # Needs a YES/STOP token somewhere; prefer it in the final sentence
        if not ("yes" in lower or "stop" in lower or "haan" in lower):
            last_char = "" if body.endswith(("?", "!", ".")) else "."
            body += f"{last_char} Reply YES if you want me to go ahead — STOP se opt out."
            fix = "appended missing YES/STOP tokens"
    elif cta == "open_ended":
        # Best practice: end with a question mark when feasible
        last_q_pos = lower.rfind("?")
        if last_q_pos == -1:
            body = body.rstrip(".!") + "?"
            fix = "converted ending into question"

    result["body"] = body
    return fix


# ─────────────────────────────────────────────────────────────────────
# Numeric provenance — anti-hallucination for cited numbers
# ─────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"₹\s?[0-9][0-9,]*|[0-9][0-9,]{1,}")
_BENIGN_TOKENS = {"2026", "2025", "1800"}   # years / common constants


def _flat_text(obj) -> str:
    """Flatten any context object into lowercase searchable text without
    thousand separators (so '2,410' matches 2410 in body)."""
    if isinstance(obj, (dict, list)):
        return _json.dumps(obj, ensure_ascii=False).replace(",", "").lower()
    return str(obj).replace(",", "").lower()


def provenance_issues(body: str, *contexts) -> list[str]:
    """
    Return issues for every number >=100 cited in the body that cannot be
    traced back to any provided context object. Years are exempt.
    Tolerates comma-separator differences between body and JSON.
    """
    hay = " ".join(_flat_text(c) for c in contexts if c)
    body_flat = body.replace(",", "")
    problems: list[str] = []
    seen = set()
    for raw in _NUMBER_RE.findall(body_flat):
        tok = raw.replace("₹", "").strip()
        if not tok.isdigit():
            continue
        n = int(tok)
        if n < 100 or tok.lower() in _BENIGN_TOKENS or tok.lower() in seen:
            continue
        seen.add(tok.lower())
        if tok.lower() not in hay:
            problems.append(f"cited number '{raw}' not found in any provided context")
    return problems


def strip_unverified_numbers(
    body: str,
    *contexts,
) -> tuple[str, list[str]]:
    """Deterministic final net: remove sentences citing unverifiable numbers.
    Returns (cleaned_body, removed_descriptions)."""
    problems = provenance_issues(body, *contexts)
    if not problems:
        return body, []

    # Extract the offending tokens once ("₹199" -> "199")
    bad_tokens = []
    for p in problems:
        token = p.split("'")[1]
        probe = token.replace("₹", "").replace(",", "").strip()
        if probe.isdigit():
            bad_tokens.append(probe)
    if not bad_tokens:
        return body, []

    sentences = re.split(r"(?<=[.!?\n])\s+", body)
    kept, removed = [], []
    for s in sentences:
        s_probe = s.replace("₹", "").replace(",", "")
        offender = None
        for probe in bad_tokens:
            if re.search(rf"(?<!\d){re.escape(probe)}(?!\d)", s_probe):
                offender = probe
                break
        if offender is not None:
            removed.append(f"dropped sentence citing unverified number {offender}")
            continue
        kept.append(s)

    cleaned = " ".join(k for k in kept if k.strip()).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned, removed


def repair_composition(
    result: dict,
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    provenance_contexts: tuple = (),
) -> tuple[dict, list[str]]:
    """
    Deterministic repair of common LLM failures. Applied as final safety net
    so zero flawed messages leave the pipeline. Returns (result, repairs_log).
    `provenance_contexts` are raw context objects for number-traceability checks.
    """
    repairs = []
    body = result.get("body", "")

    # 1. Strip markdown leakage
    cleaned = strip_markdown(body)
    if cleaned != body:
        repairs.append("stripped markdown artifacts")
        body = cleaned

    # 2. Remove category-taboo phrases
    taboos = category.get("voice", {}).get("vocab_taboo", [])
    body, removed = remove_taboos(body, taboos)
    if removed:
        repairs.append(f"removed taboo phrases: {removed}")

    # 3. Fix send_as mismatches
    scope = trigger.get("scope", "merchant")
    send_as = result.get("send_as", "")
    expected = "merchant_on_behalf" if (scope == "customer" and customer) else "vera"
    if send_as != expected:
        result["send_as"] = expected
        repairs.append(f"fixed send_as '{send_as}' -> '{expected}'")

    # 4. Fix invalid cta enum
    if result.get("cta") not in ("open_ended", "binary_yes_stop", "none"):
        result["cta"] = "binary_yes_stop" if scope == "merchant" else "none"
        repairs.append(f"invalid cta reset to '{result['cta']}'")

    result["body"] = body

    # 5. Enforce CTA placement/shape
    cta_fix = enforce_cta_last(result, trigger.get("kind", ""))
    if cta_fix:
        repairs.append(cta_fix)

    # 6. Numeric provenance final net — drop sentences citing unverifiable
    #    numbers (only when we wouldn't gut the message doing so)
    if provenance_contexts:
        cleaned, dropped = strip_unverified_numbers(body, *provenance_contexts)
        if dropped:
            # Safety: don't ship a gutted message
            if len(cleaned.strip()) >= 30:
                body = cleaned
                repairs.extend(dropped)
            else:
                repairs.append("unverified numbers detected but message too short to auto-trim")

    result["body"] = body
    return result, repairs
