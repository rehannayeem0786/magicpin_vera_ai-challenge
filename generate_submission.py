#!/usr/bin/env python3
"""
Generate submission.jsonl — runs the composer on all 30 test pairs
and writes the required submission file.

Usage:
    1. First generate the expanded dataset:
       python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded

    2. Then generate the submission:
       python generate_submission.py
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from composer import EngagementComposer


async def main():
    dataset_dir = Path("dataset/expanded")

    # Check if expanded dataset exists
    if not dataset_dir.exists():
        print("Expanded dataset not found. Generating...")
        os.system("python dataset/generate_dataset.py --seed-dir dataset --out dataset/expanded")

    # Check for test_pairs.json
    test_pairs_path = dataset_dir / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"ERROR: {test_pairs_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    # Load all data
    print("Loading dataset...")

    # Categories
    categories = {}
    cat_dir = dataset_dir / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            data = json.load(open(f))
            categories[data.get("slug", f.stem)] = data

    # Merchants
    merchants = {}
    merch_dir = dataset_dir / "merchants"
    if merch_dir.exists():
        for f in merch_dir.glob("*.json"):
            data = json.load(open(f))
            merchants[data.get("merchant_id", f.stem)] = data

    # Customers
    customers = {}
    cust_dir = dataset_dir / "customers"
    if cust_dir.exists():
        for f in cust_dir.glob("*.json"):
            data = json.load(open(f))
            customers[data.get("customer_id", f.stem)] = data

    # Triggers
    triggers = {}
    trig_dir = dataset_dir / "triggers"
    if trig_dir.exists():
        for f in trig_dir.glob("*.json"):
            data = json.load(open(f))
            triggers[data.get("id", f.stem)] = data

    # Test pairs
    with open(test_pairs_path) as f:
        test_pairs = json.load(f).get("pairs", [])

    print(f"Loaded: {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")
    print(f"Test pairs: {len(test_pairs)}")

    # Compose messages for each test pair
    composer = EngagementComposer()
    results = []

    for i, pair in enumerate(test_pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"\n[{i + 1}/{len(test_pairs)}] {test_id}: {trigger_id}")

        trigger = triggers.get(trigger_id)
        merchant = merchants.get(merchant_id)
        customer = customers.get(customer_id) if customer_id else None

        if not trigger:
            print(f"  WARNING: Trigger {trigger_id} not found, skipping")
            continue
        if not merchant:
            print(f"  WARNING: Merchant {merchant_id} not found, skipping")
            continue

        cat_slug = merchant.get("category_slug", "")
        category = categories.get(cat_slug)
        if not category:
            print(f"  WARNING: Category {cat_slug} not found, skipping")
            continue

        try:
            result = await composer.compose(category, merchant, trigger, customer)
            safe_body = result.get('body', '')[:80].encode('ascii', 'replace').decode('ascii')
            print(f"  [OK] Body ({len(result.get('body', ''))} chars): \"{safe_body}...\"")
            print(f"    CTA: {result.get('cta')}, Send as: {result.get('send_as')}")

            results.append({
                "test_id": test_id,
                "body": result.get("body", ""),
                "cta": result.get("cta", "open_ended"),
                "send_as": result.get("send_as", "vera"),
                "suppression_key": result.get("suppression_key", ""),
                "rationale": result.get("rationale", ""),
            })
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            results.append({
                "test_id": test_id,
                "body": f"Error composing: {e}",
                "cta": "none",
                "send_as": "vera",
                "suppression_key": "",
                "rationale": f"Composition failed: {e}",
            })

    # Write submission.jsonl
    output_path = Path("submission.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Submission written to {output_path}")
    print(f"Total entries: {len(results)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
