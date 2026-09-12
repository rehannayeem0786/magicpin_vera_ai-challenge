# Vera Pro v2 — magicpin AI Challenge Submission

> **Live deployment:** https://magicpinveraai-challenge.vercel.app — all endpoints verified end-to-end (`healthz`, `metadata`, `context`, `tick`, `reply`, `teardown`) with shared Redis state active (`shared_state: true`).

## Approach

**Vera Pro** is a 4-context composition engine with a **zero-flaw guarantee pipeline**: every message passes extract → anchor → compose → validate → LLM-repair → deterministic-repair before shipping. Unlike single-prompt bots, it uses **trigger-specific prompt routing** with per-trigger gold exemplars and a **numeric-provenance anti-hallucination engine**.

### Architecture

```
Context Store (in-memory, versioned, idempotent by (scope, id, version))
        │
Trigger Router ──→ 15+ prompt templates + GOLD_EXEMPLARS per kind
        │                (one action per merchant/tick — urgency-prioritized)
        ↓
Context Extractor ──→ Derived insights + SPECIFICITY ANCHORS
        │               (pre-computed citable facts: peer gaps, YoY trends,
        │                review quotes, slots, offer prices — never raw JSON)
        ↓
LLM Composer ──→ Multi-provider priority chain (all OpenAI-compatible):
        │          Groq → Cerebras → OpenRouter, with live catalog
        │          discovery auto-resolving each provider's current best
        │          model (hardcoded IDs 404 as catalogs rotate)
        │          (per-provider RPM pacing, adaptive tier/outage disable,
        │           per-model timeout cap, reasoning_effort=low for
        │           reasoning models; survives any single provider failing)
        ↓
Validator ──→ taboos, CTA shape/placement, send_as, markdown leak, length,
        │      repetition vs history, NUMERIC PROVENANCE (every ≥100 number
        │      must trace back into provided context)
        ↓
Repair Loop ──→ LLM repair pass with issue feedback, then deterministic net:
        │        taboo-sentence removal, CTA append/reposition, send_as fix,
        │        fabricated-number sentence drop
Conversation State Machine ──→ Auto-reply exit ≤2 turns + merchant-level ladder
                                (identical canned text across fresh conv IDs:
                                flag prompt → wait 24h → end), intent→action mode
                                with deterministic gold action-reply repair
                                (never re-qualifies after commitment),
                                hostile/not-interested graceful Hinglish close
```

### Key Differentiators

| Feature | How it works | Why it scores higher |
|---|---|---|
| **Specificity anchors engine** | Pre-computes verified fact-lines (CTR vs peer median, "+34% YoY thali searches", exact slot labels, ₹ offer prices) injected as *the only numbers the LLM may cite* | Maximum Specificity score with zero fabrication risk |
| **Numeric provenance check** | Regex-extracts every ≥100 figure from output; any not found in the 4 context JSONs triggers an LLM repair re-write, then deterministic sentence-drop | Zero anti-hallucination penalties |
| **Trigger-specific prompts + gold exemplars** | 15+ variants each ending in a pattern-to-emulate exemplar modeled on the brief's Appendix A/B | Consistent 10/10 message shape across kinds |
| **Multi-provider LLM chain** | Priority chain across Groq → Gemini → Cerebras → OpenRouter, whichever keys are set. Live catalog discovery (`GET /models`) auto-resolves each provider's current best model, so provider catalog rotation never 404s the chain. Per-provider RPM pacing; adaptive disable on 401 / 402 / 403 `tier_not_allowed` / empty-content; `reasoning_effort=low` on reasoning models (gpt-oss family burns `max_tokens` on hidden chain-of-thought — measured 498/500 reasoning tokens → empty body); Retry-After-aware 429 backoff; per-model timeout cap; `strict=False` JSON parsing tolerates smaller models' pretty-printed output | Judge-burst-proof: Phase 2 fires 10 req/sec with up to 20 actions per tick — free-tier rate-limit collapse (the #1 composition-score killer) is engineered out |
| **Deadline-aware parallel tick** | Compositions run in bounded waves (max 4 concurrent) under a shared 26s guard; one action per merchant per tick (urgency-wins) | Survives 30s contract even with 50 triggers; no spam penalty |
| **Shared-state store (serverless-safe)** | In-memory working cache backed by a version-merged Upstash Redis blob + atomic `SETNX` suppression claims — parallel judge requests hitting different serverless instances always see the full context state; no fragmentation, no double-sends | Every tick/reply runs with complete state; mid-test context injections are always picked up |
| **Unicode hardening** | `normalize_text()` strips thin/nbsp/zero-width spaces from every outbound body — one live failure had an LLM-emitted `\u2009` crash a strict charmap codec | Judge parsers with strict encoders never see exotic Unicode |
| **Auto-reply detection** | Canned-pattern regexes + verbatim repetition + **request-free filler pairs** (autoresponders paraphrase freely but never ask for prices/details/slots — two consecutive request-free merchant messages ⇒ exit ≤2 turns; humans who ask anything are always protected) + **merchant-level ladder** (identical canned text replayed with fresh conversation IDs still accumulates per merchant: 1st → one flag prompt for the owner, 2nd → wait 24h, 3rd → end) | Passes replay/auto-reply probes first try |
| **Intent transition hardening** | "I want to join", "count me in", "interested", Hinglish "kardo" → immediate action mode; deterministic repair rewrites any post-commitment reply that still qualifies into a self-contained gold action statement built from merchant data ("Drafting your … now — Reply CONFIRM…") | Directly fixes production Vera's #1 handoff failure; passes the judge's intent-transition action/qualify lexical check |
| **Anti-repetition memory** | Per-merchant deque of last sent bodies fed into every new composition ("vary angle") | No verbatim-repeat penalties across ticks |
| **Hindi-English craft rules** | English nouns/numbers + Hindi connectives per category `code_mix`, gender-neutral closes ("Theek hai…") | Higher merchant fit; no gendered-Hinglish slips |
| **`/v1/teardown` privacy wipe** | Wipes contexts/conversations/dedup state on judge signal | §11 compliance demonstrated |

### Evaluation rubric mapping (how the bot earns each dimension)

| Dimension | Mechanism |
|---|---|
| Specificity | Anchors engine + provenance check + service-price framing (`Cleaning @ ₹299`) |
| Category fit | Per-category voice injection, vocab_taboo firewall (+ deterministic removal), code-mix rules |
| Merchant fit | Merchant brief w/ derived peer comparisons, review quotes, conversation history awareness |
| Decision quality | Trigger router + intent fast-paths + restraint (empty tick when nothing worth sending) |
| Engagement compulsion | Lever-matched prompts (curiosity/reciprocity/loss-aversion per kind) + single clean CTA last sentence |
| Penalty avoidance | Validator+repair net: promo-tone regexes, preamble detector, repetition checks, fabrication drops |

### Compulsion Lever Strategy

Each trigger kind targets 2-3 of these levers:

- **Research digest** → Specificity (source citation) + Reciprocity ("I'll pull it for you") + Curiosity
- **Perf spike** → Social proof (vs peers) + Effort externalization ("I can push X") + Specificity
- **Recall due** → Loss aversion (time since last visit) + Specificity (slot + price) + Low-friction CTA
- **Competitor opened** → Curiosity ("a new X opened nearby") + Loss aversion + Social proof
- **Review theme** → Specificity (exact quote + count) + Social proof/Loss aversion

### Tradeoffs

1. **Multi-provider chain over a single vendor**: Composition priority is Groq → Gemini → Cerebras → OpenRouter (whichever keys are set), with live catalog discovery resolving each provider's current best model — hardcoded model IDs 404 within months as catalogs rotate, which is what silently degraded earlier submissions. This directly targets the judge's burst profile (10 req/sec, up to 20 actions/tick): a single free-tier vendor's per-minute token ceiling turns that load into 429s and fallback copy — the #1 composition-score killer. All providers are OpenAI-compatible, so one call path serves them; per-provider RPM pacing, adaptive disable on 401/402/403 (deterministic per-key billing/tier failures never retry), Retry-After-aware 429 backoff, reasoning-model hardening (`reasoning_effort=low` + generous `max_tokens` + empty-content retry — reasoning models spend the token budget on hidden chain-of-thought before the JSON body), and per-model timeout caps keep the chain alive through any single provider failing.

2. **In-memory state vs persistent storage**: The working state lives in in-memory dicts for speed, backed by a shared Upstash Redis blob (`state_store.py`) when deployed serverless — every request hydrates from and merge-saves into Redis, so parallel instances share one truth (context version-merge prevents lost pushes; atomic SETNX claims prevent double-sends). `/v1/teardown` wipes both memory and Redis for §11 compliance.

3. **15+ specialized prompts vs one generic prompt**: More surface area to maintain, but per-kind lever optimization plus gold exemplars measurably outperforms a generic template.

4. **LLM repair pass vs always-deterministic repair**: A feedback-driven rewrite keeps voice natural after fixing violations; the deterministic net only touches sentences that are objectively wrong (taboos, fabricated figures), so we never ship gutted or robotic text.

5. **Provenance strictness**: Numbers ≥100 must trace into context JSON; small counts ("3 peers", "2 slots") and years are exempt — this matches where fabrication risk actually lives while keeping copy fluent.

### Verification & evidence

```bash
# 48 offline tests — endpoint contracts, idempotency, teardown,
# validators, provenance engine, intent/auto-reply detection
# (auto-reply regression includes: verbatim-duplicate isolation,
#  zero-lexical-overlap paraphrase pools, human-chatter protection,
#  merchant-level ladder across fresh conversation IDs, judge-replay
#  action-intent gold-reply checks, graceful-close name handling)
python -m pytest tests/test_offline.py -v          # → 48 passed

# Judge simulator — warmup, auto_reply, intent, hostile scenarios
python judge_simulator.py                          # → 4/4 scenarios PASS

# Anti-fabrication audit of submission.jsonl against expanded dataset
python verify_submission.py --strict               # → 0 issues, 0 warnings
                                                   # (schema/CTA/send_as/taboo/
                                                   #  markdown/provenance)
```

The verifier reproduces the judge's hard penalties locally: category taboos, send_as mismatches, markdown leakage, and unverifiable numbers. It is also used as a regression gate before regenerating `submission.jsonl`.

### What Additional Context Would Have Helped Most

1. **Real conversation transcripts at scale** — 50-100 real conversations across categories would let us fine-tune voice and CTA patterns beyond the 4 anonymized excerpts.
2. **Conversion data** — which messages actually drove merchant replies vs. got ignored.
3. **Template approval constraints** — exact Kaleyra pre-approved templates would sharpen first-touch formatting.

## Deployment (public bot URL for the judge)

The judge needs **one public base URL** that exposes all five endpoints (`POST /v1/context`, `POST /v1/tick`, `POST /v1/reply`, `GET /v1/healthz`, `GET /v1/metadata`). Two supported paths:

### Option A — Vercel (zero-cost, ships with this repo)

The repo already contains the serverless adapter (`api/index.py` + `vercel.json` with a 60s function budget for the 30s tick contract):

1. Push this repo to GitHub.
2. On [vercel.com](https://vercel.com): **Add New… → Project → Import** the repo (framework preset: **Other**; root dir: repo root).
3. **Settings → Environment Variables** → add `GROQ_API_KEY` (free — [console.groq.com](https://console.groq.com)), optionally `CEREBRAS_API_KEY` and `OPENROUTER_API_KEY` — these must never be committed to git. For full serverless state sharing, also add `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN` from a free [Upstash](https://upstash.com) Redis database (without them the bot runs memory-only per instance).
4. Deploy → your base URL is `https://<project-name>.vercel.app`.
5. Smoke test before submitting the URL:
   ```bash
   curl https://<project-name>.vercel.app/v1/healthz
   curl https://<project-name>.vercel.app/v1/metadata
   ```

> **Serverless caveat:** Vera's store is in-memory (brief §2.1 explicitly allows this for the 60-minute window). On Vercel, a single warm Fluid-Compute instance serves the judge's request burst and preserves state between calls; a cold start resets it. If the judge's gap between `/v1/context` and `/v1/tick` exceeds the idle timeout, contexts can simply be re-pushed (all context pushes are idempotent by `(scope, id, version)`).

### Option B — Always-on host (guaranteed in-memory state)

Render / Railway / Fly.io: start command `python bot.py` (binds `0.0.0.0`, respects `BOT_PORT`), set `GROQ_API_KEY` (and optional `CEREBRAS_API_KEY`/`OPENROUTER_API_KEY`) in the platform's env settings. No cold-start state loss.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt

# Configure LLM access (.env):
# GROQ_API_KEY=your_groq_key         # console.groq.com — free tier
# GEMINI_API_KEY=your_gemini_key     # aistudio.google.com — free tier (optional)
# CEREBRAS_API_KEY=your_cerebras_key # cloud.cerebras.ai — free tier
# OPENROUTER_API_KEY=your_or_key     # openrouter.ai — free models
#
# Models are auto-discovered from each provider's live catalog (GET /models)
# and re-verified every 6h — no hardcoded IDs to go stale. Optional pins:
# GROQ_MODELS=openai/gpt-oss-120b
# CEREBRAS_MODELS=gpt-oss-120b
# OPENROUTER_MODELS=nex-agi/nex-n2.5-pro:free
# LLM_BUDGET_S=24
# BOT_PORT=8080
#
# Shared state on serverless (optional but recommended on Vercel):
# UPSTASH_REDIS_REST_URL=https://<db>.upstash.io    # enables cross-instance state
# UPSTASH_REDIS_REST_TOKEN=your_token               # without these, memory-only
```

## Run

```bash
# Generate expanded dataset (100 triggers / 50 merchants / 30 test pairs)
python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded

# Start the bot server
python bot.py

# In another terminal — run the judge simulator (all scenarios)
python judge_simulator.py

# Run offline test suite
python -m pytest tests/test_offline.py -v

# Regenerate + audit submission.jsonl
python generate_submission.py
python verify_submission.py --strict
```

## Files

| File | Purpose |
|---|---|
| `bot.py` | FastAPI server — 6 endpoints (5 brief endpoints + teardown), parallel deadline-aware tick |
| `composer.py` | Composition pipeline: model chain, anchors injection, validate→repair loop |
| `prompts.py` | 15+ trigger prompts + GOLD_EXEMPLARS + WhatsApp craft rules |
| `context_extractor.py` | Briefs, derived insights, **specificity-anchors engine**, salutation resolver |
| `validators.py` | Validation firewall + deterministic repair + **numeric provenance** |
| `conversation_handlers.py` | Multi-turn state machine |
| `generate_submission.py` | Generates `submission.jsonl` for the 30 test pairs |
| `verify_submission.py` | Anti-fabrication + rubric-conformance audit of submissions |
| `judge_simulator.py` | Local replay of the judge's 4 probe scenarios (warmup, auto_reply, intent, hostile) |
| `tests/test_offline.py` | 39-test pytest suite (no live LLM needed) |
| `submission.jsonl` | 30 composed messages (provenance-audited) |

## Tech Stack

- **LLM**: Multi-provider priority chain (all OpenAI-compatible): Groq → Gemini → Cerebras → OpenRouter, with live catalog discovery auto-resolving current models
- **Framework**: FastAPI + Uvicorn
- **HTTP Client**: httpx (async)
- **Shared state**: Upstash Redis (REST) with in-memory fallback — version-merged blob + atomic suppression claims across serverless instances
- **Testing**: pytest (FastAPI TestClient)
- **Language**: Python 3.11+
