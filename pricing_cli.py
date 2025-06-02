#!/usr/bin/env python3
"""Command-line interface for the two-stage dynamic pricing system.

Usage
-----
python pricing_cli.py --inventory 100 --price 100.0 --bids bids.csv [--alpha 0.05]

The *bids* file should contain either:
1) two comma-separated columns  customer_id,bid_value  (header optional), or
2) a single column of bid values (one per row).  In the second case customers
   are implicitly numbered 0,1,2,… in their order of appearance.

Outputs the required metrics:
    T/C           proportion of customers with fulfilled orders (expected)
    T/N           proportion of items sold (expected)
    Per-customer  outcome: bid paid (Stage-1), personalised price (Stage-2),
                  or "Incomplete" if not offered a unit.
    P×T           target revenue (face_price × expected units sold)
    Remaining bal difference between expected revenue and target revenue.

The script reuses the algorithm implemented in *dynamic_pricing_system.py*.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
from typing import List, Tuple, Dict

import numpy as np

# Local import (same directory)
from dynamic_pricing_system import stage1_allocation, stage2_pricing  # type: ignore

# Local replica of the acceptance-probability function (same logic as inside
# dynamic_pricing_system).  Kept here to avoid changing the core module.
def acceptance_prob(P_i: float, B_i: float) -> float:  # noqa: N802  (keep original name)
    if P_i <= B_i:
        return 1.0
    elif P_i <= 2 * B_i:
        return ((2 * B_i - P_i) / B_i) ** 2
    else:
        return 0.0


def read_bids(path: pathlib.Path) -> List[Tuple[int, float]]:
    """Read bids file and return list[(customer_id, bid)]."""
    bids: List[Tuple[int, float]] = []
    with path.open(newline="") as fp:
        reader = csv.reader(fp)
        # Detect and skip header if non-numeric in second column
        first_row_peek = next(reader)
        try:
            float(first_row_peek[-1])  # attempt parse last column as float
            # It was numeric -> treat as data
            rows = [first_row_peek] + list(reader)
        except ValueError:
            # Header present -> skip
            rows = list(reader)
        for idx, row in enumerate(rows):
            if len(row) == 0:
                continue
            if len(row) == 1:
                # single column: bid value only
                cid = idx
                bid_val = float(row[0])
            else:
                cid = int(row[0]) if row[0].strip() else idx
                bid_val = float(row[1])
            bids.append((cid, bid_val))
    return bids


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run two-stage pricing on a set of bids")
    parser.add_argument("--inventory", "-N", type=int, required=True, help="Total inventory (divisible by 10)")
    parser.add_argument("--price", "-P", type=float, required=True, help="Face price P")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--bids", "-b", type=pathlib.Path, help="CSV/txt file with bids")
    group.add_argument("--random", "-r", type=int, metavar="C", help="Generate C random bids in the legal range (requires --price)")
    parser.add_argument("--priority", choices=["units", "gap"], default="units", help="Stage-2 relaxation priority (units or gap)")

    args = parser.parse_args(argv)

    N: int = args.inventory
    P: float = args.price

    if args.bids is None and args.random is None:
        parser.error("Provide either --bids FILE or --random C to generate bids")

    if args.bids:
        bids = read_bids(args.bids)
    else:
        rng = np.random.default_rng(42)
        C_rand = args.random
        if C_rand is None or C_rand <= 0:
            parser.error("--random must be followed by a positive integer")
        # Draw bids away from ALL bucket boundaries to guarantee eligibility.
        gap_ratio = 0.005   # 0.5 % of P away from edges
        lower = np.array([0.51,0.61,0.71,0.81,0.91,1.01,1.11,1.21,1.31,1.41])
        upper = np.array([0.60,0.70,0.80,0.90,1.00,1.10,1.20,1.30,1.40,1.50])
        bids = []
        for i in range(C_rand):
            bucket_idx = rng.integers(0,10)
            lo = lower[bucket_idx] + gap_ratio
            hi = upper[bucket_idx] - gap_ratio
            ratio = rng.uniform(lo, hi)
            bids.append((i, float(ratio * P)))
    C = len(bids)

    # --- Stage 1 ---------------------------------------------------
    accepted_stage1, remaining, t, delta1 = stage1_allocation(bids, P, N)
    print(f"Stage-1 surplus / deficit Δ1: {delta1:+.2f} € (positive = surplus)")

    inventory_left = N - t

    # ------------------------------------------------------------------
    # If Stage 1 has already exhausted the stock we can stop here: the
    # MILP would be void and Stage 2 cannot adjust revenue anyway.
    # ------------------------------------------------------------------
    if inventory_left <= 0:
        print("No inventory left after Stage 1. All units were sold in Stage 1.")

        T_exp = float(t)  # deterministic — all 100 % of Stage-1 acceptances
        prop_customers = T_exp / C if C else 0.0
        prop_inventory = T_exp / N if N else 0.0
        target_revenue = P * T_exp
        vendor_revenue_exp = sum(b for _, b in accepted_stage1)
        remaining_balance = vendor_revenue_exp - target_revenue

        print("===== SUMMARY =====")
        print(f"T/C (customers served): {prop_customers:.3f}")
        print(f"T/N (items sold)     : {prop_inventory:.3f}")
        print()

        print("Per-customer outcome:")
        for cid, bid_val in bids:
            if cid in {x for x, _ in accepted_stage1}:
                print(f"Customer {cid:>3}: BidPaid:{bid_val:.2f}")
            else:
                print(f"Customer {cid:>3}: Incomplete")

        print("\n===== FINANCIALS =====")
        print(f"Face price × T (target revenue): €{target_revenue:.2f}")
        print(f"Expected vendor revenue        : €{vendor_revenue_exp:.2f}")
        print(f"Remaining balance              : €{remaining_balance:+.2f}")
        return  # nothing more to do

    # --- Stage 2 ---------------------------------------------------
    prices_stage2, lambda_star = stage2_pricing(remaining, P, delta1, inventory_left, priority=args.priority)

    # Build quick lookup tables
    accepted_ids = {cid for cid, _ in accepted_stage1}
    price_dict: Dict[int, float] = {cid: price for cid, price in prices_stage2}

    # Compute expected Stage-2 acceptances & revenue
    expected_acceptances = 0.0
    stage2_rev = 0.0
    for (cid, price), (_, bid_val) in zip(prices_stage2, remaining):
        f_i = acceptance_prob(price, bid_val)
        expected_acceptances += f_i
        stage2_rev += price * f_i

    # Cap expected Stage-2 acceptances by physical stock left
    expected_acceptances_capped = min(expected_acceptances, inventory_left)
    T_exp = t + expected_acceptances_capped  # expected total units sold (capped)

    # Metrics -------------------------------------------------------
    prop_customers = T_exp / C if C else 0.0
    prop_inventory = T_exp / N if N else 0.0
    target_revenue = P * T_exp
    vendor_revenue_exp = sum(b for _, b in accepted_stage1) + stage2_rev
    remaining_balance = vendor_revenue_exp - target_revenue

    # --- Output ----------------------------------------------------
    print("===== SUMMARY =====")
    print(f"T/C (customers served): {prop_customers:.3f}")
    print(f"T/N (items sold)     : {prop_inventory:.3f}")
    print()

    print("Per-customer outcome:")
    for cid, bid_val in bids:
        if cid in accepted_ids:
            print(f"Customer {cid:>3}: BidPaid:{bid_val:.2f}")
        elif cid in price_dict:
            price = price_dict[cid]
            prob = acceptance_prob(price, bid_val)

            EPS = 1e-6  # tolerance for floating-point noise
            if prob < EPS:
                print(f"Customer {cid:>3}: Incomplete")
            elif prob > 1.0 - EPS:
                # Treat probabilities extremely close to 1 as certain sales
                print(f"Customer {cid:>3}: Sold:{price:.2f}")
            else:
                print(f"Customer {cid:>3}: Offer:{price:.2f} (p={prob*100:.0f}%)")
        else:
            print(f"Customer {cid:>3}: Incomplete")

    print("\n===== FINANCIALS =====")
    print(f"Face price × T (target revenue): €{target_revenue:.2f}")
    print(f"Expected vendor revenue        : €{vendor_revenue_exp:.2f}")
    print(f"Remaining balance              : €{remaining_balance:+.2f}")


if __name__ == "__main__":
    main() 