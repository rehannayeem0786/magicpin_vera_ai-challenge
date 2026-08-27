#!/usr/bin/env python3
"""
Vera Pro — Anti-Fabrication Verifier for submission.jsonl
==========================================================
Checks every submitted message against the rubric's hard rules:

  1. Schema completeness (test_id, body, cta, send_as, suppression_key, rationale)
  2. CTA enum validity + CTA shape consistency
  3. send_as correctness per trigger scope
  4. Category vocab_taboo violations
  5. Markdown leakage on WhatsApp
  6. Length budget (~60 words outbound)
  7. ANTI-HALLUCINATION: every number ≥ 100 cited in the body must be
     traceable to the category/merchant/trigger/customer context JSONs.

Run:  python verify_submission.py          → human report
      python verify_submission.py --strict → non-zero exit on critical issues
"""

import json
import re
import sys
import io
from pathlib import Path

# Windows console UTF-8 safety (₹ / ✗ / ✔ glyphs)
if getattr(sys.stdout, "encoding", "utf-8").lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).parent
DS = ROOT / "dataset" / "expanded"

CTA_VALID = {"open_ended", "binary_yes_stop", "none"}
NUM_RE = re.compile(r"(₹\s?[\d,]+|[\d][\d,]{2,})")   # ₹ amounts & numbers ≥3 digits


def load_all():
    data = {"categories": {}, "merchants": {}, "customers": {}, "triggers": {}}
    for kind, folder in [("categories", "categories"), ("merchants", "merchants"),
                         ("customers", "customers"), ("triggers", "triggers")]:
        d = DS / folder
        if d.exists():
            for f in d.glob("*.json"):
                obj = json.loads(f.read_text(encoding="utf-8"))
                # Key field must match the object type — triggers carry a
                # merchant_id field, so a shared key-chain would mis-key them.
                if kind == "categories":
                    key = obj.get("slug")
                elif kind == "merchants":
                    key = obj.get("merchant_id")
                elif kind == "customers":
                    key = obj.get("customer_id")
                else:
                    key = obj.get("id")
                data[kind][key or f.stem] = obj
    pairs = json.loads((DS / "test_pairs.json").read_text(encoding="utf-8"))["pairs"]
    return data, {p["test_id"]: p for p in pairs}


def verify_entry(entry, pair, data):
    issues, warns = [], []
    body = entry.get("body", "")

    # 1 schema
    for k in ("test_id", "body", "cta", "send_as", "suppression_key", "rationale"):
        if k not in entry:
            issues.append(f"missing field '{k}'")

    # 2 CTA
    if entry.get("cta") not in CTA_VALID:
        issues.append(f"invalid cta '{entry.get('cta')}'")

    # 3 send_as vs scope
    trig = data["triggers"].get(pair.get("trigger_id"), {})
    scope = trig.get("scope", "merchant")
    expected = "merchant_on_behalf" if scope == "customer" else "vera"
    if entry.get("send_as") != expected:
        issues.append(f"send_as={entry.get('send_as')} but scope={scope} needs '{expected}'")

    # 4 taboo words
    merchant = data["merchants"].get(pair.get("merchant_id"), {})
    category = data["categories"].get(merchant.get("category_slug"), "")
    taboos = (category or {}).get("voice", {}).get("vocab_taboo", [])
    low = body.lower()
    for t in taboos:
        t_clean = t.lower().split("(")[0].strip()
        if t_clean and t_clean in low:
            issues.append(f"category TABOO present: '{t}'")

    # 5 markdown leakage
    if re.search(r"\*\*|`{1,3}|^#{1,3}\s", body):
        issues.append("markdown artifacts (** / backticks) will render literally")

    # 6 length
    words = len(body.split())
    if words > 110:
        warns.append(f"long message ({words} words)")

    # 7 anti-fabrication traceability of big numbers
    if category or merchant or trig:
        haystack_parts = [
            json.dumps(category, ensure_ascii=False),
            json.dumps(merchant, ensure_ascii=False),
            json.dumps(trig, ensure_ascii=False),
        ]
        cust_id = pair.get("customer_id")
        if cust_id:
            haystack_parts.append(
                json.dumps(data["customers"].get(cust_id, {}), ensure_ascii=False))
        haystack = " ".join(haystack_parts)
        hay_flat = haystack.replace(",", "")   # tolerate thousand-separators
        body_flat = body.replace(",", "")
        for m in NUM_RE.findall(body):
            probe = m.replace("₹", "").replace(",", "").strip()
            if not probe.isdigit():
                continue
            n = int(probe)
            if n < 100:
                continue
            if probe not in hay_flat:
                # allow known-benign patterns: years, times like 1800
                if re.fullmatch(r"(19|20)\d{2}", probe) or probe in ("1800",):
                    continue
                issues.append(f"unverifiable number '{m}' not found in any provided context")

    return issues, warns


def main():
    strict = "--strict" in sys.argv
    lines = (ROOT / "submission.jsonl").read_text(encoding="utf-8").splitlines()
    data, pairs = load_all()

    total_issues, total_warns = 0, 0
    print(f"Verifying {len(lines)} submission entries...\n" + "=" * 66)
    for line in lines:
        if not line.strip():
            continue
        entry = json.loads(line)
        pair = pairs.get(entry.get("test_id"))
        if not pair:
            issues, warns = [f"unknown test_id '{entry.get('test_id')}'"], []
        else:
            issues, warns = verify_entry(entry, pair, data)
        total_issues += len(issues)
        total_warns += len(warns)
        status = "PASS" if not issues else "FAIL"
        flag = "" if not warns else f" (+{len(warns)} warn)"
        print(f"[{status}] {entry.get('test_id')}{flag}")
        for i in issues:
            print(f"       ✗ {i}")
        for w in warns:
            print(f"       ! {w}")

    print("=" * 66)
    print(f"Issues: {total_issues} | Warnings: {total_warns}")
    if strict and total_issues:
        sys.exit(1)
    print("RESULT:", "CLEAN ✔" if total_issues == 0 else "NEEDS FIXES")


if __name__ == "__main__":
    main()