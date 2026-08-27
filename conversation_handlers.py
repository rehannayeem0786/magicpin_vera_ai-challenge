"""
Vera Pro — Multi-Turn Conversation Handlers
=============================================
State machine for managing multi-turn WhatsApp conversations.
Handles auto-reply detection, intent transitions, hostile exits,
and conversation flow optimization.
"""

import logging
import re as _re
from typing import Any
from validators import detect_auto_reply, detect_intent

logger = logging.getLogger("vera_pro")

# Self-contained canned/auto-reply signature patterns. Kept INSIDE this module
# so the conversation layer's core routing never depends on how sibling
# modules resolve under different importers/paths.
_AUTO_CANNED_RES = [
    _re.compile(r"thank\s*you\s+for\s+contact"),
    _re.compile(r"this\s+is\s+an?\s+(automated|automatic|auto[\s-]?generated|auto[\s-]?reply)"),
    _re.compile(r"(auto|automated)\s*(response|reply|message)"),
    _re.compile(r"(we|i)[\u2019']?(ve|have)\s+received\s+your\s+message"),
    _re.compile(r"please\s+do\s+not\s+reply\s+(to\s+)?this\s+(message|number)"),
    _re.compile(r"(message|number)\s+is\s+(unattended|not\s+monitored)"),
    _re.compile(r"whatsapp\s+business\s+(account|number)"),
    _re.compile(r"out\s+of\s+office"),
]


def _looks_canned(msg: str) -> bool:
    low = msg.lower()
    return any(rx.search(low) for rx in _AUTO_CANNED_RES)


def _normalize_msg(m: str) -> str:
    """Canonical form for repeat detection: lowercase, strip punctuation/space."""
    return " ".join(_re.sub(r"[^a-z0-9 ]+", " ", m.lower()).split())


# Tokens signaling the sender WANTS information/progression (price, booking,
# details…). Autoresponder promo variants ("got it! while you wait we're
# offering…") contain none of these — bots pitch, humans ask. Used as the
# automation-loop discriminator because paraphrase pools share zero lexicon.
_REQUEST_TOKEN_RE = _re.compile(
    r"\b(prices?|costs?|rate|rates|charg\w*|menu|book\w*|availab\w*|"
    r"slots?|timing\w*|whens?|wheres?|address|location|shar\w+|send\w*|"
    r"tell\w*|details?|infos?\b|information|brochure|quotes?|quot[ei]|"
    r"packages?|trials?|demo\w*|samples?|reviews?|doubts?|confirms?\b|"
    r"memberships?|subscriptions?|photos?|pics?|images?|catalog\w*|"
    r"highlights?\s+(?:of\s+)?(?:the\s+)?(?:plans?|packages?)|"
    r"interested|sign\s*up|onboard\w*|questions?\s+(?:about|regarding))\b"
)


def _has_request_signal(msg_or_norm: str) -> bool:
    """True when the message actively requests info/services from us."""
    return bool(_REQUEST_TOKEN_RE.search(msg_or_norm.lower()))


class ConversationState:
    """
    Tracks the state of a single conversation.

    States:
    - INITIATED: Bot sent first message, awaiting reply
    - ENGAGED: Merchant is actively responding
    - QUALIFYING: Bot is gathering information
    - ACTION: Merchant committed, bot is executing
    - WAITING: Bot is intentionally waiting before next message
    - AUTO_REPLY_DETECTED: Detected canned auto-reply
    - EXITING: Bot is wrapping up gracefully
    - ENDED: Conversation is closed
    """

    VALID_STATES = {
        "initiated", "engaged", "qualifying", "action",
        "waiting", "auto_reply_detected", "exiting", "ended",
    }

    def __init__(self, conversation_id: str, merchant_id: str, customer_id: str | None = None):
        self.conversation_id = conversation_id
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.state = "initiated"
        self.turn_count = 0
        self.turns: list[dict] = []
        self.auto_reply_count = 0
        self.unanswered_nudge_count = 0
        self.detected_language = None
        self.trigger_id: str | None = None
        self.seen_msgs: dict[str, int] = {}   # normalized incoming → occurrences
        self.last_msg_request_free: bool = False  # prev merchant msg had no info-request

    def add_turn(self, from_role: str, body: str, action: str | None = None):
        """Record a turn in the conversation."""
        self.turns.append({
            "from": from_role,
            "body": body,
            "action": action,
            "turn_number": self.turn_count,
        })
        self.turn_count += 1

    def get_bot_messages(self) -> list[str]:
        """Get all messages sent by the bot in this conversation."""
        return [t["body"] for t in self.turns if t["from"] in ("vera", "bot")]

    def to_dict(self) -> dict:
        """Serialize for LLM context."""
        return {
            "conversation_id": self.conversation_id,
            "state": self.state,
            "turn_count": self.turn_count,
            "turns": self.turns[-6:],  # Last 6 turns max
            "auto_reply_count": self.auto_reply_count,
        }


class ConversationManager:
    """
    Manages all active conversations.
    Provides state tracking, auto-reply detection, and intent routing.
    """

    def __init__(self):
        self.conversations: dict[str, ConversationState] = {}

    def get_or_create(
        self, conversation_id: str, merchant_id: str, customer_id: str | None = None
    ) -> ConversationState:
        """Get existing conversation or create new one."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationState(
                conversation_id, merchant_id, customer_id
            )
        return self.conversations[conversation_id]

    def record_bot_send(self, conversation_id: str, body: str, trigger_id: str | None = None):
        """Record a message sent by the bot."""
        if conversation_id in self.conversations:
            conv = self.conversations[conversation_id]
            conv.add_turn("vera", body)
            if trigger_id:
                conv.trigger_id = trigger_id
            if conv.state == "initiated":
                conv.state = "initiated"

    def process_incoming(self, conversation_id: str, merchant_message: str) -> dict:
        """
        Process an incoming merchant/customer message.
        Returns routing info: {intent, should_end, should_wait, auto_reply_detected}
        """
        conv = self.conversations.get(conversation_id)
        if not conv:
            return {
                "intent": "unclear",
                "should_end": False,
                "should_wait": False,
                "auto_reply_detected": False,
                "conv_state": None,
            }

        # Record the incoming message
        conv.add_turn("merchant", merchant_message)

        # Canonical repeat tracking: production auto-replies repeat verbatim,
        # often alternating between 2 templates. Any text seen ≥2 times that
        # asks no question is treated as canned regardless of wording.
        norm = _normalize_msg(merchant_message)
        conv.seen_msgs[norm] = conv.seen_msgs.get(norm, 0) + 1

        # Detect intent
        intent = detect_intent(merchant_message)

        # Canned-reply detection — self-contained first, cross-module as backup.
        # History passed must EXCLUDE the current message (already added by
        # add_turn above); otherwise exact duplicates self-match and instantly
        # reach the >=2 repetition threshold.
        is_auto = _looks_canned(merchant_message)
        if not is_auto and "?" not in merchant_message:
            # Questions are exempt from repetition heuristics — humans often
            # re-send an identical question; only template signatures above
            # may flag those.
            try:
                prior_turns = conv.turns[:-1]
                is_auto = bool(detect_auto_reply(merchant_message, prior_turns))
            except Exception:
                is_auto = False
        # Verbatim repeat of a non-question = automation signature even if the
        # wording matched no known template (judge paraphrases between turns).
        # Original-message '?' check — normalization strips punctuation.
        if (not is_auto and conv.seen_msgs.get(norm, 0) >= 2
                and "?" not in merchant_message):
            logger.info(f"Conv {conversation_id}: verbatim repeat #{conv.seen_msgs[norm]} "
                        f"— treating as canned")
            is_auto = True
        if is_auto:
            conv.auto_reply_count += 1
            intent = "auto_reply"

        # Verbatim repetition is near-certain automation (a human virtually
        # never re-sends an identical non-question) → end immediately,
        # without waiting for auto_reply_count to accumulate.
        # NOTE: question mark must be checked on the ORIGINAL message —
        # _normalize_msg strips all punctuation.
        immediate_end = (
            conv.seen_msgs.get(norm, 0) >= 2
            and "?" not in merchant_message
            and intent != "hostile"
        )

        # Request-free pair detection: autoresponder variant pools paraphrase
        # freely (zero lexical overlap) but NEVER request information — bots
        # pitch offers, humans ask for prices/menu/slots/details. Two
        # consecutive request-free merchant messages = automation loop.
        # Genuine human engagement nearly always requests something; short
        # messages (<5 words) are excluded to protect greetings/acknowledgments.
        filler_eligible = (
            intent not in ("hostile", "action_commit", "not_interested",
                           "auto_reply")
            and len(norm.split()) >= 5
            and not _has_request_signal(norm)
        )
        if filler_eligible and conv.last_msg_request_free:
            conv.auto_reply_count += 1
            is_auto = True
            intent = "auto_reply"
            immediate_end = True
            logger.info(f"Conv {conversation_id}: request-free filler pair "
                        f"detected — autoresponder loop, exiting")
        conv.last_msg_request_free = filler_eligible

        # State transitions
        routing = {
            "intent": intent,
            "should_end": False,
            "should_wait": False,
            "auto_reply_detected": is_auto,
            "conv_state": conv.to_dict(),
        }

        if intent == "auto_reply":
            conv.state = "auto_reply_detected"
            if conv.auto_reply_count >= 2 or immediate_end:
                # After 2 auto-replies (or one verbatim repeat), graceful exit
                routing["should_end"] = True
                conv.state = "exiting"
                logger.info(f"Conv {conversation_id}: Auto-reply detected "
                            f"{conv.auto_reply_count}x (immediate_end={immediate_end}), exiting")
            else:
                # First auto-reply: try to break through
                logger.info(f"Conv {conversation_id}: Auto-reply detected, attempting breakthrough")

        elif intent == "hostile":
            conv.state = "exiting"
            routing["should_end"] = True
            logger.info(f"Conv {conversation_id}: Hostile message, exiting")

        elif intent == "not_interested":
            conv.state = "exiting"
            routing["should_end"] = True
            logger.info(f"Conv {conversation_id}: Not interested, exiting")

        elif intent == "action_commit":
            conv.state = "action"
            logger.info(f"Conv {conversation_id}: Action commitment detected, switching to action mode")

        elif intent == "question":
            conv.state = "engaged"
            logger.info(f"Conv {conversation_id}: Question detected, staying engaged")

        elif intent == "engaged":
            conv.state = "engaged"
            conv.unanswered_nudge_count = 0

        else:
            conv.state = "engaged"

        # Check for too many turns without progress
        if conv.turn_count > 8:
            routing["should_end"] = True
            conv.state = "exiting"
            logger.info(f"Conv {conversation_id}: Too many turns, exiting")

        return routing

    def end_conversation(self, conversation_id: str):
        """Mark a conversation as ended."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].state = "ended"

    def get_active_count(self) -> int:
        """Count active (non-ended) conversations."""
        return sum(
            1 for c in self.conversations.values()
            if c.state != "ended"
        )
