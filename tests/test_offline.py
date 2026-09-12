"""
Vera Pro — Offline Endpoint & Unit Tests
=========================================
No LLM calls: verifies HTTP contracts, idempotency rules, teardown privacy,
validators, intent/auto-reply detection, and deterministic repairs.

Run:  .venv\\Scripts\\python.exe -m pytest tests/test_offline.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient  # noqa: E402


def _client():
    import bot
    bot.contexts.clear()
    bot.conversations_mgr.conversations.clear()
    bot.used_suppression_keys.clear()
    bot.sent_bodies.clear()
    bot.autoresponder_memory.clear()
    return TestClient(bot.app)


# ─────────────────────────────────────────────────────────────────────
# HTTP contract tests (5 endpoints per testing brief §2)
# ─────────────────────────────────────────────────────────────────────

class TestHealthAndMetadata:
    def test_healthz_shape(self):
        c = _client()
        r = c.get("/v1/healthz")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data and "contexts_loaded" in data

    def test_metadata_identity(self):
        c = _client()
        r = c.get("/v1/metadata")
        assert r.status_code == 200
        d = r.json()
        assert d["team_name"] and d["model"] and d["approach"]


class TestContextEndpoint:
    def _push(self, c, scope="merchant", cid="m_001", version=1):
        return c.post("/v1/context", json={
            "scope": scope, "context_id": cid, "version": version,
            "payload": {"test": True}, "delivered_at": "2026-04-26T10:00:00Z",
        })

    def test_accepts_valid_push(self):
        c = _client()
        r = self._push(c)
        assert r.status_code == 200
        assert r.json()["accepted"] is True
        assert "ack_id" in r.json() and "stored_at" in r.json()

    def test_idempotent_same_version(self):
        c = _client()
        self._push(c, version=3)
        r = self._push(c, version=3)  # re-post same version → no-op accept
        assert r.status_code == 200
        assert r.json()["accepted"] is True

    def test_stale_version_rejected(self):
        c = _client()
        self._push(c, version=5)
        r = self._push(c, version=3)  # older → stale_version
        body = r.json()
        assert body["accepted"] is False
        assert body["reason"] == "stale_version"
        assert body["current_version"] == 5

    def test_higher_version_replaces_atomically(self):
        c = _client()
        self._push(c, cid="dentists", scope="category", version=1)
        self._push(c, cid="dentists", scope="category", version=7)
        stored = None
        for (scope, cid), entry in __import__("bot").contexts.items():
            if scope == "category" and cid == "dentists":
                stored = entry["version"]
        assert stored == 7

    def test_invalid_scope_rejected(self):
        c = _client()
        r = self._push(c, scope="galaxy")
        assert r.json()["accepted"] is False
        assert r.json()["reason"] == "invalid_scope"


class TestTickContract:
    def test_empty_tick_returns_empty_actions_immediately(self):
        c = _client()
        r = c.post("/v1/tick", json={
            "now": "2026-04-26T10:30:00Z", "available_triggers": [],
        })
        assert r.status_code == 200
        assert r.json()["actions"] == []

    def test_unknown_triggers_skipped_gracefully(self):
        c = _client()
        r = c.post("/v1/tick", json={
            "now": "2026-04-26T10:30:00Z",
            "available_triggers": ["trg_nonexistent_1"],
        })
        assert r.status_code == 200
        assert r.json()["actions"] == []


class TestTeardownPrivacy:
    def test_teardown_wipes_all_state(self):
        import bot as bot_mod
        c = _client()
        c.post("/v1/context", json={
            "scope": "merchant", "context_id": "m_x", "version": 1,
            "payload": {"secret": True}, "delivered_at": "2026-04-26T10:00:00Z",
        })
        r = c.post("/v1/teardown")
        assert r.status_code == 200 and r.json()["wiped"] is True
        assert len(bot_mod.contexts) == 0


# ─────────────────────────────────────────────────────────────────────
# Validators — anti-pattern firewall
# ─────────────────────────────────────────────────────────────────────

class TestValidators:
    def test_taboo_detection(self):
        from validators import validate_composition
        result = {
            "body": "This is a guaranteed packed house offer!",
            "cta": "open_ended", "send_as": "vera",
        }
        cat = {"slug": "restaurants",
               "voice": {"vocab_taboo": ["guaranteed packed house"]}}
        ok, issues = validate_composition(
            result, cat, {}, {"scope": "merchant", "kind": "x"}, None,
        )
        assert any("TABOO" in i for i in issues)

    def test_send_as_mismatch_for_customer_scope(self):
        from validators import validate_composition
        result = {
            "body": "Hi Priya, reminder about your cleaning.",
            "cta": "binary_yes_stop", "send_as": "vera",
        }
        cat, cust = {"voice": {}}, {"identity": {"name": "Priya"}}
        ok, issues = validate_composition(
            result, cat, {}, {"scope": "customer"}, cust,
        )
        assert any("SEND_AS" in i for i in issues)

    def test_markdown_leak_flagged(self):
        from validators import validate_composition
        result = {
            "body": "Dental Cleaning at **₹299** — book now?",
            "cta": "open_ended", "send_as": "vera",
        }
        ok, issues = validate_composition(
            result, {"voice": {}}, {}, {"scope": "merchant"}, None,
        )
        assert any("MARKDOWN" in i for i in issues)


class TestDeterministicRepair:
    def test_strip_markdown_removes_bold(self):
        from validators import strip_markdown
        assert strip_markdown("**Dental Cleaning** @ ₹299") == "Dental Cleaning @ ₹299"

    def test_taboo_sentence_removed(self):
        from validators import remove_taboos
        body, removed = remove_taboos(
            "Miracle marketing for you. Also try our thali.", ["miracle marketing"],
        )
        assert "miracle marketing" not in body.lower()
        assert "thali" in body

    def test_binary_cta_appended_when_missing(self):
        from validators import enforce_cta_last
        result = {"cta": "binary_yes_stop", "body": "Offer ready for you."}
        enforce_cta_last(result)
        assert "YES" in result["body"] and "STOP" in result["body"]

    def test_open_ended_ends_with_question(self):
        from validators import enforce_cta_last
        result = {"cta": "open_ended", "body": "Want me to draft it."}
        enforce_cta_last(result)
        assert result["body"].endswith("?")


class TestNumericProvenance:
    def test_verified_number_passes(self):
        from validators import provenance_issues
        ctx = ({"performance": {"views": 2410}}, {}, {}, None)
        issues = provenance_issues("Your listing got 2,410 views this month!", *ctx)
        assert issues == []

    def test_fabricated_price_flagged(self):
        from validators import provenance_issues
        issues = provenance_issues(
            "Try our Keratin @ ₹199 this month!",
            {"offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]},
        )
        assert any("₹199" in i for i in issues)

    def test_strip_unverified_removes_only_offending_sentence(self):
        from validators import strip_unverified_numbers
        body = ("Ravi, thali searches up 34% near you. "
                "Try our invented package @ ₹199. Want a draft?")
        ctx = ({"trend_signals": [{"query": "thali", "delta_yoy": 0.34}]},)
        cleaned, dropped = strip_unverified_numbers(body, *ctx)
        assert "34%" in cleaned          # verified stat survives
        assert "₹199" not in cleaned     # fabricated price gone
        assert len(dropped) == 1

    def test_years_exempt(self):
        from validators import provenance_issues
        assert provenance_issues("Recall set up in 2026 for you.", {}) == []


# ─────────────────────────────────────────────────────────────────────
# Intent & auto-reply detection (conversation layer)
# ─────────────────────────────────────────────────────────────────────

class TestIntentDetection:
    def test_action_commit(self):
        from validators import detect_intent
        for msg in ["Ok lets do it", "Yes please send details", "Haan kardo",
                    "I want to join", "go ahead"]:
            assert detect_intent(msg) == "action_commit", msg

    def test_hostile(self):
        from validators import detect_intent
        for msg in ["Stop messaging me", "Don't contact me again"]:
            assert detect_intent(msg) == "hostile", msg

    def test_not_interested(self):
        from validators import detect_intent
        for msg in ["not interested", "No thanks", "nahi chahiye"]:
            assert detect_intent(msg) == "not_interested", msg

    def test_question(self):
        from validators import detect_intent
        assert detect_intent("What is the price?") == "question"

    def test_auto_reply_canned(self):
        from validators import detect_auto_reply
        msg = "Thank you for contacting us. We will get back to you soon."
        assert detect_auto_reply(msg) is True

    def test_auto_reply_repeat_exits_faster(self):
        from validators import detect_auto_reply
        history = [{"from": "merchant", "body": "This is an automated message from ABC Salon."}]
        dup = "This is an automated message from ABC Salon."
        # Same canned text twice → detected on 2nd occurrence (exit ≤ turn 3)
        assert detect_auto_reply(dup, history + history) is True

    def test_single_verbatim_dup_not_validators_job(self):
        """Repetition policy lives in ConversationManager (canonical + immediate
        exit). A single prior occurrence must NOT flag here — otherwise every
        short non-question self-matches via its own history entry."""
        from validators import detect_auto_reply
        history = [{"from": "merchant",
                    "body": "Please send me the pricing details and packages"}]
        assert detect_auto_reply(history[0]["body"], history) is False

    def test_repeated_question_never_auto_flagged(self):
        from conversation_handlers import ConversationManager
        mgr = ConversationManager()
        mgr.get_or_create("conv_dq", "m_test")
        msg = "Please send me the pricing details and packages?"
        for _ in range(3):
            r = mgr.process_incoming("conv_dq", msg)
            assert r["intent"] != "auto_reply"
            assert not r["should_end"]


class TestConversationAutoReplyEnd:
    """Simulates the judge harness playing canned auto-replies."""

    def test_alternating_templates_end_within_2_turns(self):
        from conversation_handlers import ConversationManager
        mgr = ConversationManager()
        cid = "conv_ar_test"
        msgs = [
            "Hi! This is an automated response from Glow Salon.",
            "We have received your message and will respond shortly.",
            "Hi! This is an automated response from Glow Salon.",   # verbatim repeat
        ]
        # Production contract: conversations are registered when the bot sends
        # its proactive message (tick); /v1/reply then lands on an existing conv.
        conv = mgr.get_or_create(cid, "m_test")
        self._drive(mgr, cid, msgs, conv)

    def _drive(self, mgr, cid, msgs, conv):
        outcomes = []
        for i, m in enumerate(msgs, 1):
            r = mgr.process_incoming(cid, m)
            outcomes.append((i, r["intent"], r["should_end"]))
            if r["should_end"]:
                break
        assert any(e for _, _, e in outcomes), f"never ended: {outcomes}"
        assert len(outcomes) <= 2, f"exited too slowly (brief: ≤2 turns): {outcomes}"

    def test_unique_human_messages_never_end(self):
        from conversation_handlers import ConversationManager
        mgr = ConversationManager()
        cid = "conv_human"
        mgr.get_or_create(cid, "m_test")
        for m in ["what does this cost?",
                  "is evening slot available?",
                  "share menu options",
                  "sounds good lets do it"]:
            r = mgr.process_incoming(cid, m)
            assert r["intent"] != "auto_reply", m
            assert not r["should_end"], m

    def test_paraphrased_autoresponder_exits_by_turn_2(self):
        """Judge harness generates a DIFFERENT boilerplate variant each turn.
        Two overlapping non-question replies in a row = autoresponder cycle."""
        from conversation_handlers import ConversationManager
        mgr = ConversationManager()
        cid = "conv_para"
        mgr.get_or_create(cid, "m_test")
        msgs = [
            "Thank you for your message! Meanwhile, would you like to know about our weekend offers?",
            "Got it! While you wait, did you know we're offering exciting deals this week?",
        ]
        ended_at = None
        for i, m in enumerate(msgs, 1):
            r = mgr.process_incoming(cid, m)
            if r["should_end"]:
                ended_at = i
                break
        assert ended_at == 2, f"expected exit at turn 2, got {ended_at}"

    def test_judge_observed_zero_overlap_pool_exits(self):
        """Exactly the variant pool observed in a judge run — messages share
        almost no tokens, so only the request-free-filler rule catches them."""
        from conversation_handlers import ConversationManager
        mgr = ConversationManager()
        cid = "conv_pool"
        mgr.get_or_create(cid, "m_test")
        pool = [
            "Got it! Meanwhile, would you like to highlight our weekend offers to more customers?",
            "Great! While you wait, did you know we\u2019re offering exciting deals this week?",
            "Sure! Meanwhile, would you like to highlight our weekend offers to more customers?",
        ]
        ended_at = None
        for i, m in enumerate(pool, 1):
            r = mgr.process_incoming(cid, m)
            if r["should_end"]:
                ended_at = i
                break
        assert ended_at is not None and ended_at <= 3, f"no exit by turn 3 ({ended_at})"

    def test_request_signal_tokens_protect_humans(self):
        from conversation_handlers import _has_request_signal
        positives = ["what are your rates?", "please share price list",
                     "can we book a slot tomorrow", "send me details",
                     "is table available tonight", "share the menu"]
        for p in positives:
            assert _has_request_signal(p), p
        negatives = ["got it great amazing wonderful",
                     "thank you so much really appreciate",
                     "we are offering exciting deals this week"]
        for n in negatives:
            assert not _has_request_signal(n), n

    def test_low_overlap_human_chatter_stays_engaged(self):
        from conversation_handlers import ConversationManager
        mgr = ConversationManager()
        cid = "conv_chatter"
        mgr.get_or_create(cid, "m_test")
        for m in ["please share the salon price list today",
                  "my friend recommended your bridal package",
                  "ok send booking details for friday"]:
            r = mgr.process_incoming(cid, m)
            if not r["should_end"]:
                continue
            raise AssertionError(f"ended early on human chatter: {m!r}")


class TestSalutationResolution:
    def test_restaurant_template_resolved(self):
        from context_extractor import build_salutation
        cat = {"voice": {"salutation_examples": [
            "Hi {chef_or_owner_first_name}", "{restaurant_name} team"]}}
        merch = {"identity": {"owner_first_name": "Ravi",
                              "name": "Cafe Delhi"}}
        assert build_salutation(cat, merch) == "Hi Ravi"

    def test_no_unresolved_braces(self):
        from context_extractor import build_salutation
        cat = {"voice": {"salutation_examples": ["Namaste {business_name}!"]}}
        merch = {"identity": {"owner_first_name": "", "name": "Glow Salon"}}
        assert "{" not in build_salutation(cat, merch)


class TestSpecificityAnchors:
    def test_anchors_only_from_real_data(self):
        from context_extractor import extract_specificity_anchors
        cat = {
            "peer_stats": {"avg_ctr": 0.030},
            "trend_signals": [{"query": "biryani near me",
                               "delta_yoy": 0.18}],
        }
        merch = {
            "performance": {"views": 2410, "calls": 18, "directions": 45,
                            "ctr": 0.021, "leads": 9},
            "subscription": {"status": "active", "plan": "Pro",
                             "days_remaining": 82},
            "offers": [{"title": "Dental Cleaning @ ₹299",
                        "status": "active"}],
        }
        joined = "\n".join(
            extract_specificity_anchors(cat, merch, {}, None))
        assert "2,410 views" in joined
        assert "82 days remaining" in joined
        assert "Dental Cleaning @ ₹299" in joined

    def test_slot_anchor_for_customer_trigger(self):
        from context_extractor import extract_specificity_anchors
        trigger = {"kind": "recall_due", "payload": {"available_slots": [
            {"label": "Wed 5 Nov, 6pm"}, {"label": "Thu 6 Nov, 5pm"}]}}
        customer = {"relationship": {
            "last_visit": "2026-05-12T00:00:00Z", "visits_total": 4}}
        joined = "\n".join(
            extract_specificity_anchors({}, {}, trigger, customer))
        assert "Wed 5 Nov, 6pm" in joined
        assert "lifetime visits" in joined


# ─────────────────────────────────────────────────────────────────────
# Judge-replay scenarios (brief Example 4.1 / 4.2) — offline & deterministic
# ─────────────────────────────────────────────────────────────────────

AUTO_MSG = "Thank you for contacting us! Our team will respond shortly."
COMMIT_MSG = "Ok lets do it. Whats next?"

# Exactly the lexical contract judge_simulator.py applies to replies
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]
ACTIONING = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]


def _force_offline(monkeypatch):
    """Make the composer deterministic (no LLM) regardless of env keys."""
    import composer
    monkeypatch.setattr(composer, "CALL_PLAN", [])


def _post_reply(c, conv_id, mid, message, turn):
    return c.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": mid,
        "from_role": "merchant",
        "message": message,
        "received_at": "2026-04-26T10:00:00Z",
        "turn_number": turn,
    })


class TestMerchantAutoReplyLadder:
    """Example 4.1: judge replays the SAME canned auto-reply with a fresh
    conversation_id every turn (conv_auto_1..4). Per-conversation counts
    never accumulate — the merchant-level ladder must: flag → wait 24h → end."""

    def test_flag_then_wait_then_end_across_fresh_convs(self, monkeypatch):
        from validators import detect_auto_reply
        assert detect_auto_reply(AUTO_MSG) is True  # preconditions

        c = _client()
        _force_offline(monkeypatch)
        actions = []
        for i in range(1, 5):
            r = _post_reply(c, f"conv_auto_{i}", "m_ladder", AUTO_MSG, i + 1)
            assert r.status_code == 200
            actions.append(r.json())

        assert actions[0]["action"] == "send"
        assert "auto-reply" in actions[0]["body"].lower()

        assert actions[1]["action"] == "wait"
        assert actions[1]["wait_seconds"] == 86400

        assert actions[2]["action"] == "end"
        # 4th replay (if it ever came) stays ended
        assert actions[3]["action"] == "end"

    def test_human_question_never_triggers_ladder(self, monkeypatch):
        c = _client()
        _force_offline(monkeypatch)
        for i in range(1, 4):
            r = _post_reply(c, f"conv_human_{i}", "m_ladder2",
                            "What is the price for the bridal package?",
                            i + 1)
            data = r.json()
            assert data["action"] != "end", data
            assert data["action"] == "send", data
        import bot
        assert bot.autoresponder_memory.get("m_ladder2", {}) == {}

    def test_distinct_canned_texts_count_separately(self, monkeypatch):
        """Two DIFFERENT canned texts = still 1st occurrence each → flag
        prompts, not a premature exit."""
        c = _client()
        _force_offline(monkeypatch)
        r1 = _post_reply(c, "cv1", "m_l3",
                         "Thank you for contacting us! We will respond shortly.", 2)
        r2 = _post_reply(c, "cv2", "m_l3",
                         "Thanks for your message! Our team will reply soon.", 3)
        assert r1.json()["action"] == "send"
        assert r2.json()["action"] == "send"


class TestActionIntentGoldReply:
    """Example 4.2: after an explicit commitment the reply must be an
    action statement (actioning word, no qualifying phrase) — exactly what
    the judge simulator checks on the intent_transition scenario."""

    def test_commitment_via_composer_fallback(self, monkeypatch):
        c = _client()
        _force_offline(monkeypatch)
        r = _post_reply(c, "conv_intent_1", "m_acts", COMMIT_MSG, 2)
        assert r.status_code == 200
        data = r.json()
        assert data["action"] == "send"
        body_lower = data["body"].lower()
        assert any(w in body_lower for w in ACTIONING), data["body"]
        assert not any(w in body_lower for w in QUALIFYING), data["body"]
        assert "confirm" in body_lower
        assert data["cta"] == "binary_confirm_cancel"

    def test_self_contained_from_merchant_data(self, monkeypatch):
        c = _client()
        _force_offline(monkeypatch)
        c.post("/v1/context", json={
            "scope": "merchant", "context_id": "m_acts2", "version": 1,
            "payload": {
                "identity": {"name": "Studio11", "owner_first_name": "Lakshmi"},
                "offers": [{"title": "Bridal Trial @ ₹999", "status": "active"}],
            },
            "delivered_at": "2026-04-26T10:00:00Z",
        })
        r = _post_reply(c, "conv_intent_2", "m_acts2", COMMIT_MSG, 3)
        data = r.json()
        assert data["action"] == "send"
        assert "Lakshmi" in data["body"]
        assert "Bridal Trial @ ₹999" in data["body"]
        assert "CONFIRM" in data["body"]

    def test_deterministic_repair_when_llm_qualifies(self, monkeypatch):
        import bot
        c = _client()
        _force_offline(monkeypatch)

        async def _qualifying_llm(conv_state, message, category, merchant):
            return {"action": "send",
                    "body": "Great! Would you like me to proceed with the details?",
                    "cta": "open_ended",
                    "rationale": "LLM slipped back into qualifying"}

        monkeypatch.setattr(bot.composer, "compose_reply", _qualifying_llm)
        r = _post_reply(c, "conv_intent_3", "m_acts3", COMMIT_MSG, 3)
        data = r.json()
        body_lower = data["body"].lower()
        assert any(w in body_lower for w in ACTIONING), data["body"]
        assert not any(w in body_lower for w in QUALIFYING), data["body"]


class TestReplyGracefulCloses:
    """The judge's hostile / intent / question scenarios often run standalone
    with NO merchant context pushed — name-bearing reply templates must
    degrade cleanly (no dangling " ." from an empty name)."""

    def test_hostile_close_without_merchant_context(self, monkeypatch):
        import bot
        c = _client()
        _force_offline(monkeypatch)
        r = _post_reply(c, "cv_h1", "m_noc",
                        "Get lost. Don't message me again.", 2)
        data = r.json()
        assert data["action"] == "send"
        assert data["cta"] == "none"
        assert data["body"].startswith("Sorry")
        assert " ." not in data["body"], data["body"]
        # ended conversations are cleaned up from the live map
        assert bot.conversations_mgr.conversations.get("cv_h1") is None

    def test_not_interested_close_without_merchant_context(self, monkeypatch):
        c = _client()
        _force_offline(monkeypatch)
        r = _post_reply(c, "cv_n1", "m_noc", "Not interested, please stop", 2)
        data = r.json()
        assert data["body"].startswith("Koi baat nahi")
        assert " ." not in data["body"], data["body"]

    def test_question_fallback_without_merchant_context(self, monkeypatch):
        c = _client()
        _force_offline(monkeypatch)
        r = _post_reply(c, "cv_q1", "m_noc", "What are your weekend timings?", 2)
        data = r.json()
        assert data["action"] == "send"
        assert " ." not in data["body"], data["body"]
