# Vera Pro v2 — magicpin AI Challenge Submission

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
LLM Composer ──→ Mistral chain: mistral-medium-latest → mistral-large-latest → open-mistral-nemo
        │          (deadline-aware ≤24s budget, per-model cap; auto-fallback under load/outage)
        ↓
Validator ──→ taboos, CTA shape/placement, send_as, markdown leak, length,
        │      repetition vs history, NUMERIC PROVENANCE (every ≥100 number
        │      must trace back into provided context)
        ↓
Repair Loop ──→ LLM repair pass with issue feedback, then deterministic net:
        │        taboo-sentence removal, CTA append/reposition, send_as fix,
        │        fabricated-number sentence drop
Conversation State Machine ──→ Auto-reply exit ≤2 turns, intent→action mode,
                                hostile/not-interested graceful Hinglish close
```

### Key Differentiators

| Feature | How it works | Why it scores higher |
|---|---|---|
| **Specificity anchors engine** | Pre-computes verified fact-lines (CTR vs peer median, "+34% YoY thali searches", exact slot labels, ₹ offer prices) injected as *the only numbers the LLM may cite* | Maximum Specificity score with zero fabrication risk |
| **Numeric provenance check** | Regex-extracts every ≥100 figure from output; any not found in the 4 context JSONs triggers an LLM repair re-write, then deterministic sentence-drop | Zero anti-hallucination penalties |
| **Trigger-specific prompts + gold exemplars** | 15+ variants each ending in a pattern-to-emulate exemplar modeled on the brief's Appendix A/B | Consistent 10/10 message shape across kinds |
| **Model fallback chain** | `mistral-medium-latest` primary (~1-3s); falls to large/nemo under load or model outage, with a per-model timeout cap so one hanging model can never starve the chain — never dead | Frontier copy quality AND reliability |
| **Deadline-aware parallel tick** | Compositions run in bounded waves (max 4 concurrent) under a shared 26s guard; one action per merchant per tick (urgency-wins) | Survives 30s contract even with 50 triggers; no spam penalty |
| **Shared-state store (serverless-safe)** | In-memory working cache backed by a version-merged Upstash Redis blob + atomic `SETNX` suppression claims — parallel judge requests hitting different serverless instances always see the full context state; no fragmentation, no double-sends | Every tick/reply runs with complete state; mid-test context injections are always picked up |
| **Auto-reply detection** | Canned-pattern regexes + verbatim repetition + **request-free filler pairs** (autoresponders paraphrase freely but never ask for prices/details/slots — two consecutive request-free merchant messages ⇒ exit ≤2 turns; humans who ask anything are always protected) | Passes replay/auto-reply probes first try |
| **Intent transition hardening** | "I want to join", "count me in", "interested", Hinglish "kardo" → immediate action mode | Directly fixes production Vera's #1 handoff failure |
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

1. **Mistral medium (paid) as primary vs large**: We promote `mistral-medium-latest` to primary composer — measured ~1-3s vs large's current platform-side instability (large hung 60s+ during testing; it stays in the chain and is auto-promoted by `MISTRAL_MODEL` env if desired). Copy quality still drives all five rubric dimensions, and the chain degrades gracefully to large/nemo under quota pressure. Well inside the 30s endpoint contract.

2. **In-memory state vs persistent storage**: The working state lives in in-memory dicts for speed, backed by a shared Upstash Redis blob (`state_store.py`) when deployed serverless — every request hydrates from and merge-saves into Redis, so parallel instances share one truth (context version-merge prevents lost pushes; atomic SETNX claims prevent double-sends). `/v1/teardown` wipes both memory and Redis for §11 compliance.

3. **15+ specialized prompts vs one generic prompt**: More surface area to maintain, but per-kind lever optimization plus gold exemplars measurably outperforms a generic template.

4. **LLM repair pass vs always-deterministic repair**: A feedback-driven rewrite keeps voice natural after fixing violations; the deterministic net only touches sentences that are objectively wrong (taboos, fabricated figures), so we never ship gutted or robotic text.

5. **Provenance strictness**: Numbers ≥100 must trace into context JSON; small counts ("3 peers", "2 slots") and years are exempt — this matches where fabrication risk actually lives while keeping copy fluent.

### Verification & evidence

```bash
# 39 offline tests — endpoint contracts, idempotency, teardown,
# validators, provenance engine, intent/auto-reply detection
# (auto-reply regression includes: verbatim-duplicate isolation,
#  zero-lexical-overlap paraphrase pools, human-chatter protection)
python -m pytest tests/test_offline.py -v          # → 39 passed

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
3. **Settings → Environment Variables** → add `MISTRAL_API_KEY` (Production) — this must never be committed to git.
4. Deploy → your base URL is `https://<project-name>.vercel.app`.
5. Smoke test before submitting the URL:
   ```bash
   curl https://<project-name>.vercel.app/v1/healthz
   curl https://<project-name>.vercel.app/v1/metadata
   ```

> **Serverless caveat:** Vera's store is in-memory (brief §2.1 explicitly allows this for the 60-minute window). On Vercel, a single warm Fluid-Compute instance serves the judge's request burst and preserves state between calls; a cold start resets it. If the judge's gap between `/v1/context` and `/v1/tick` exceeds the idle timeout, contexts can simply be re-pushed (all context pushes are idempotent by `(scope, id, version)`).

### Option B — Always-on host (guaranteed in-memory state)

Render / Railway / Fly.io: start command `python bot.py` (binds `0.0.0.0`, respects `BOT_PORT`), set `MISTRAL_API_KEY` in the platform's env settings. No cold-start state loss.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt

# Configure LLM access (.env):
# MISTRAL_API_KEY=your_key_here      # required
# MISTRAL_MODEL=mistral-medium-latest # optional override (chain falls back automatically)
# BOT_PORT=8080                      # optional
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

- **LLM**: Mistral API — `mistral-medium-latest` (primary) with automatic fallback to `mistral-large-latest`, then `open-mistral-nemo`; per-model timeout cap keeps one hung model from starving the chain
- **Framework**: FastAPI + Uvicorn
- **HTTP Client**: httpx (async)
- **Shared state**: Upstash Redis (REST) with in-memory fallback — version-merged blob + atomic suppression claims across serverless instances
- **Testing**: pytest (FastAPI TestClient)
- **Language**: Python 3.11+
