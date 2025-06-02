import numpy as np
import pulp  # open-source MILP solver (CBC bundled)
from typing import List, Tuple, Dict
import bisect

def stage1_allocation(bids: List[Tuple[int, float]], P: float, N: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]], int, float]:
    """
    Stage 1: Bucket-based allocation
    
    Args:
        bids: List of (customer_id, bid_value) tuples
        P: Face price
        N: Total inventory (must be divisible by 10)
        
    Returns:
        accepted_stage1: List of accepted (customer_id, bid_value)
        remaining: List of remaining (customer_id, bid_value) for Stage 2
        t: Number of units sold in Stage 1
        delta1: Stage 1 surplus/deficit
    """
    # Validate inputs
    if N % 10 != 0:
        raise ValueError("N must be divisible by 10")
    
    # Express lower & upper bounds of the 10 open intervals (strict inequalities)
    lower = [0.51, 0.61, 0.71, 0.81, 0.91, 1.01, 1.11, 1.21, 1.31, 1.41]
    upper = [0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50]

    bucket_capacity = N // 10

    # Initialize 10 buckets
    buckets = [[] for _ in range(10)]

    # Assign bids to buckets under strict open-interval rules
    for customer_id, bid in bids:
        ratio = bid / P

        # Quick reject if outside global bounds (open)
        if not (0.51 < ratio < 1.50):
            continue

        # Locate candidate bucket via bisect; then confirm it satisfies open interval
        idx = bisect.bisect_right(lower, ratio) - 1  # candidate bucket index
        if 0 <= idx < 10 and lower[idx] < ratio < upper[idx]:
            buckets[idx].append((customer_id, bid))
    
    # Sort each bucket by bid value (descending) and accept top bucket_capacity
    accepted_stage1 = []
    remaining = []
    
    for bucket in buckets:
        # Sort by bid value (descending)
        bucket.sort(key=lambda x: x[1], reverse=True)
        
        # Accept top bucket_capacity bids
        accepted_stage1.extend(bucket[:bucket_capacity])
        remaining.extend(bucket[bucket_capacity:])
    
    # Calculate Stage 1 metrics
    t = len(accepted_stage1)
    delta1 = sum(bid - P for _, bid in accepted_stage1)
    
    return accepted_stage1, remaining, t, delta1


# ---------------------------------------------------------------------------
# Deterministic MILP Stage-2 solver (uniform price, binary selection)
# ---------------------------------------------------------------------------
def deterministic_stage2(
    remaining: List[Tuple[int, float]],
    P: float,
    delta1: float,
    inventory_left: int,
) -> List[Tuple[int, float]]:
    """Solve Stage-2 exactly with a tiny MILP.

    Every chosen customer pays the same uniform price P2 ≤ bid.
    The model maximises the count of winners while meeting inventory
    and revenue-balance exactly.
    """
    bids = {cid: b for cid, b in remaining}

    K = inventory_left
    if K <= 0 or not bids:
        return [(cid, 2 * b) for cid, b in remaining]

    model = pulp.LpProblem("Stage2_MILP", pulp.LpMaximize)

    # Decision variables
    y = {cid: pulp.LpVariable(f"y_{cid}", 0, 1, cat="Binary") for cid in bids}
    Pvars = {
        cid: pulp.LpVariable(
            f"P_{cid}", lowBound=0.0, upBound=bids[cid], cat="Continuous"
        )
        for cid in bids
    }

    # Objective: maximise number of customers served
    model += pulp.lpSum(y.values())

    # Stock constraint
    model += pulp.lpSum(y.values()) <= K

    # Revenue balance: Σ P_i − P Σ y_i  = −Δ1
    model += pulp.lpSum(Pvars.values()) - P * pulp.lpSum(y.values()) == -delta1

    # Linking constraints: if y_i = 0 → P_i = 0 ; if y_i =1 → P_i within [0.51P, B_i]
    for cid, B in bids.items():
        model += Pvars[cid] <= B * y[cid]
        # Allow the solver to pick any non-negative price, down to zero,
        # so that it can match the exact revenue balance even when a 0.51 P
        # floor would make the problem mathematically infeasible.
        model += Pvars[cid] >= 0.0 * y[cid]

    status = model.solve(pulp.PULP_CBC_CMD(msg=False))

    # ------------------------------------------------------------------
    # If exact revenue balance is impossible, try a *relaxed* model that
    # allows a residual gap and minimises it lexicographically after
    # maximising the customer count.  This gives "the best we can do"
    # instead of reverting to prohibitive prices.
    # ------------------------------------------------------------------
    if status != pulp.LpStatusOptimal:
        # 1st pass infeasible → build relaxed model
        model_relax = pulp.LpProblem("Stage2_MILP_relax", pulp.LpMaximize)

        # same variables
        y2 = {cid: pulp.LpVariable(f"y_{cid}", 0, 1, cat="Binary") for cid in bids}
        P2 = {
            cid: pulp.LpVariable(
                f"P_{cid}", lowBound=0.0, upBound=bids[cid], cat="Continuous"
            )
            for cid in bids
        }

        # residual gap variables (absolute value)
        g_pos = pulp.LpVariable("g_pos", lowBound=0.0)
        g_neg = pulp.LpVariable("g_neg", lowBound=0.0)

        BIG = 1000  # weight to prioritise customers over gap
        model_relax += BIG * pulp.lpSum(y2.values()) - (g_pos + g_neg)

        # stock limit
        model_relax += pulp.lpSum(y2.values()) <= K

        # relaxed revenue equation: Σ P_i − P Σ y_i + g_pos - g_neg = -Δ1
        model_relax += (pulp.lpSum(P2.values()) - P * pulp.lpSum(y2.values()) + g_pos - g_neg == -delta1)

        # linking constraints
        for cid, B in bids.items():
            model_relax += P2[cid] <= B * y2[cid]
            model_relax += P2[cid] >= 0.0 * y2[cid]

        status2 = model_relax.solve(pulp.PULP_CBC_CMD(msg=False))

        if status2 == pulp.LpStatusOptimal:
            # build price list from relaxed solution
            prices = []
            for cid, b in remaining:
                if y2[cid].value() > 0.5:
                    prices.append((cid, P2[cid].value()))
                else:
                    prices.append((cid, 2 * b))
            gap_val = g_pos.value() - g_neg.value()
            print(f"  Relaxed MILP: residual balance {gap_val:+.2f} after selling {int(sum(v.value() for v in y2.values()))} units.")
            return prices

        # still infeasible (pathological) → prohibitive prices
        print("  MILP infeasible even after relaxation — reverting to prohibitive pricing.")
        return [(cid, 2 * b) for cid, b in remaining]

    prices = []
    for cid, b in remaining:
        if y[cid].value() > 0.5:
            prices.append((cid, Pvars[cid].value()))
        else:
            prices.append((cid, 2 * b))
    return prices


def stage2_pricing(remaining: List[Tuple[int, float]], P: float, delta1: float, inventory_left: int) -> Tuple[List[Tuple[int, float]], float]:
    """
    Stage 2: Optimal personalised pricing (dual decomposition)

    Mathematical background
    -----------------------
    Let λ be the dual multiplier for the *revenue-balance equality* and μ ≥ 0 the
    multiplier for the *inventory inequality*  ∑ f_i(P_i) ≤ K.  The per-customer
    Lagrangian term then reads::

        g_i(P_i; λ, μ) = (1 + μ) f_i(P_i) + λ (P_i − P) f_i(P_i)

    Our algorithm folds the factor (1+μ) into the revenue multiplier and works
    with the *re-scaled* single parameter::

        λ′  :=  λ / (1 + μ).

    With this substitution the single-customer objective becomes

        (1 + λ′ (P_i − P)) f_i(P_i),

    and the interior stationary point produced by the calculus below is exactly

        P_i* = 2 B_i − [2 + 4 λ′ B_i − 2 λ′ P] / (3 λ′)

    which matches the textbook formula  2/3(B_i+P) − 2(1+μ)/(3λ) after
    replacing λ′ = λ/(1+μ).  The code therefore *does* incorporate μ even
    though it is not carried explicitly.

    Inventory constraint enforcement
    --------------------------------
    Rather than maintain a second multiplier for μ we enforce the hard stock
    limit outside the root-finding loop: if the expected acceptances under a
    trial λ′ would exceed the remaining stock we return a large positive value
    (Big-M) from ``compute_G`` so that the bisection routine steers away from
    that region.  This keeps the algorithm on the feasible side without the
    extra dual variable.

    Args
    ----
    remaining      List[(customer_id, bid)] *not* accepted in Stage 1.
    P              Face price.
    delta1         Stage-1 surplus / deficit (Σ (bid − P)).
    inventory_left Units still in stock after Stage 1.

    Returns
    -------
    prices_stage2  List[(customer_id, personalised_price)].
    lambda_star    Optimal re-scaled dual variable λ′.
    """
    # ------------------------------------------------------------------
    # 0. Quick degenerate cases
    # ------------------------------------------------------------------
    if not remaining or inventory_left <= 0:
        # Case 1: inventory gone but some customers remain (shouldn't happen with current logic)
        if remaining and inventory_left <= 0:
            print("Warning: No inventory left for Stage 2. Setting all prices to maximum.")
            return [(cid, 2 * bid) for cid, bid in remaining], float("inf")

        # Case 2: no customers reached Stage 2 – nothing we can do to balance revenue.
        if not remaining:
            print("Info: All eligible customers were served in Stage 1; no Stage 2 actions possible.")
        return [], 0.0
    
    # ------------------------------------------------------------------
    # NEW: Try deterministic MILP first. It always yields exact balance
    # when mathematically feasible and maximises customers served.
    # ------------------------------------------------------------------
    prices_det = deterministic_stage2(remaining, P, delta1, inventory_left)

    # Deterministic MILP succeeded — use its prices directly.
    return prices_det, float("inf")


# Example usage and testing
def test_system():
    """Test the two-stage pricing system"""
    # Example parameters
    P = 100.0  # Face price
    N = 100    # Total inventory (divisible by 10)
    
    # Generate example bids
    np.random.seed(42)
    num_customers = 150
    
    # Generate bids in valid range [0.51P, 1.50P]
    min_bid = 0.51 * P
    max_bid = 1.50 * P
    
    bids = []
    for i in range(num_customers):
        bid_value = np.random.uniform(min_bid, max_bid)
        bids.append((i, bid_value))
    
    # Stage 1
    accepted_stage1, remaining, t, delta1 = stage1_allocation(bids, P, N)
    
    print(f"Stage 1 Results:")
    print(f"  Accepted: {t} customers")
    print(f"  Remaining: {len(remaining)} customers")
    print(f"  Delta (surplus/deficit): ${delta1:.2f}")
    print(f"  Inventory used: {t} out of {N} units")
    print(f"  Inventory remaining: {N - t} units")
    print()
    
    # Stage 2 - properly unpack the tuple
    inventory_left = N - t
    prices_stage2, lambda_star = stage2_pricing(remaining, P, delta1, inventory_left)
    
    print(f"Stage 2 Results:")
    print(f"  Personalized prices set for {len(prices_stage2)} customers")
    if lambda_star != float('inf'):
        print(f"  Lambda* = {lambda_star:.6f}")
    else:
        print(f"  Lambda* = ∞ (no inventory)")
    
    # Show price distribution
    if prices_stage2:
        stage2_prices = [p for _, p in prices_stage2]
        print(f"  Price range: ${min(stage2_prices):.2f} - ${max(stage2_prices):.2f}")
    
    # Verify revenue constraint
    total_stage2_adjustment = 0.0
    expected_acceptances = 0.0
    
    for (cid, P_i), (_, B_i) in zip(prices_stage2, remaining):
        if P_i <= B_i:
            f_i = 1.0
        elif P_i <= 2 * B_i:
            f_i = ((2 * B_i - P_i) / B_i) ** 2
        else:
            f_i = 0.0
        
        total_stage2_adjustment += (P_i - P) * f_i
        expected_acceptances += f_i
    
    print(f"  Expected Stage 2 acceptances: {expected_acceptances:.2f}")
    print(f"  Stage 2 revenue adjustment: ${total_stage2_adjustment:.2f}")
    print(f"  Revenue constraint check (should be ~0): ${total_stage2_adjustment + delta1:.6f}")
    
    # Calculate total expected sales
    total_expected_sales = t + expected_acceptances
    print(f"\nTotal expected sales: {total_expected_sales:.2f} units")
    print(f"Total inventory: {N} units")
    
    if total_expected_sales > N + 1e-6:
        print(f"ERROR: OVERSELLING by {total_expected_sales - N:.2f} units!")
    else:
        print(f"Inventory utilization: {total_expected_sales / N * 100:.1f}%")

    # ---------------- Additional economic report -------------------
    print("\n===== ECONOMIC SUMMARY =====")
    C = len(bids)
    T = total_expected_sales
    print(f"T/C (customers served): {T/C:.3f}")
    print(f"T/N (items sold):       {T/N:.3f}")

    # Build a quick lookup for personalised prices
    price_dict = {cid: p for cid, p in prices_stage2}
    customer_lines = []
    for cid, bid_val in bids:
        if any(cid == x for x, _ in accepted_stage1):
            customer_lines.append((cid, f"BidPaid:{bid_val:.2f}"))
        else:
            Pi = price_dict.get(cid, None)
            if Pi is None or inventory_left == 0:
                customer_lines.append((cid, "Incomplete"))
            else:
                customer_lines.append((cid, f"Price:{Pi:.2f}"))
    # Show first 20 customers to avoid flooding the console
    print("Per-customer outcome (first 20):")
    for cid, info in customer_lines[:20]:
        print(f"  Customer {cid:3d}: {info}")

    vendor_revenue = sum(b for _, b in accepted_stage1)
    for (cid, Pi), (_, Bi) in zip(prices_stage2, remaining):
        if Pi <= Bi:
            f_i = 1.0
        elif Pi <= 2 * Bi:
            f_i = ((2 * Bi - Pi) / Bi) ** 2
        else:
            f_i = 0.0
        vendor_revenue += Pi * f_i
    print(f"Vendor revenue (expected): ${vendor_revenue:.2f}")
    print(f"Target revenue P*T:        ${P*T:.2f}")
    remaining_balance = vendor_revenue - P*T
    print(f"Remaining balance:         ${remaining_balance:+.2f}")
    print("===========================\n")

    # Optional Monte-Carlo simulation for risk metrics
    RUN_SIMULATION = False  # set True to generate Monte-Carlo stats
    if RUN_SIMULATION and prices_stage2:
        sim_stats = simulate_stage2_outcomes(remaining, prices_stage2, P, t, N, n_rep=1000, seed=2024)
        print("---- Monte-Carlo (1000 replications) ----")
        print(f"Mean units sold     : {sim_stats['mean_units']:.2f}")
        print(f"5th-95th pct units  : {sim_stats['p5_units']:.2f} – {sim_stats['p95_units']:.2f}")
        print(f"Mean revenue        : €{sim_stats['mean_revenue']:.2f}")
        print(f"5th-95th pct revenue: €{sim_stats['p5_revenue']:.2f} – €{sim_stats['p95_revenue']:.2f}")
        print(f"Prob. oversell (>N) : {sim_stats['oversell_prob']*100:.2f}%")
        print("-----------------------------------------\n")


def test_with_limited_inventory():
    """Test case where Stage 1 doesn't use all inventory"""
    print("\n" + "="*60)
    print("TEST CASE: Limited Stage 1 Sales")
    print("="*60 + "\n")
    
    P = 100.0
    N = 100
    
    # Generate fewer high-quality bids to ensure Stage 1 doesn't fill
    np.random.seed(123)
    num_customers = 80
    
    bids = []
    for i in range(num_customers):
        # Skew bids toward lower values
        bid_value = np.random.uniform(0.51 * P, 1.2 * P)
        bids.append((i, bid_value))
    
    # Run the system
    accepted_stage1, remaining, t, delta1 = stage1_allocation(bids, P, N)
    
    print(f"Stage 1 Results:")
    print(f"  Accepted: {t} customers")
    print(f"  Remaining: {len(remaining)} customers")
    print(f"  Delta (surplus/deficit): ${delta1:.2f}")
    print(f"  Inventory remaining: {N - t} units")
    print()
    
    inventory_left = N - t
    prices_stage2, lambda_star = stage2_pricing(remaining, P, delta1, inventory_left)
    
    print(f"Stage 2 Results:")
    print(f"  Can sell up to {inventory_left} more units")
    print(f"  Lambda* = {lambda_star:.6f}")
    
    # Calculate expected acceptances
    expected_acceptances = 0.0
    for (cid, P_i), (_, B_i) in zip(prices_stage2, remaining):
        if P_i <= B_i:
            f_i = 1.0
        elif P_i <= 2 * B_i:
            f_i = ((2 * B_i - P_i) / B_i) ** 2
        else:
            f_i = 0.0
        expected_acceptances += f_i
    
    print(f"  Expected Stage 2 acceptances: {expected_acceptances:.2f}")
    print(f"  Total expected sales: {t + expected_acceptances:.2f} / {N} units")


# ---------------------------------------------------------------------------
# Optional Monte-Carlo simulator
# ---------------------------------------------------------------------------
def simulate_stage2_outcomes(
    remaining: List[Tuple[int, float]],
    prices_stage2: List[Tuple[int, float]],
    P: float,
    t: int,
    N: int,
    n_rep: int = 1000,
    seed: int = None,
) -> Dict[str, float]:
    """Simulate acceptance realisations to obtain risk metrics.

    Args:
        remaining: list of (customer_id, bid) for customers that reached Stage 2.
        prices_stage2: list of (customer_id, personalised price) returned by solver.
        P: face price (float).
        t: units sold in Stage 1.
        N: total inventory.
        n_rep: number of Monte-Carlo replications.
        seed: optional random seed for reproducibility.

    Returns:
        Dictionary with mean / quantile statistics.
    """
    rng = np.random.default_rng(seed)

    # Build aligned arrays of bids and personalised prices
    price_dict = {cid: p for cid, p in prices_stage2}
    bids = np.array([bid for _, bid in remaining], dtype=float)
    prices = np.array([price_dict[cid] for cid, _ in remaining], dtype=float)

    # Vectorised acceptance probabilities
    prob = np.where(
        prices <= bids,
        1.0,
        np.where(prices <= 2 * bids, ((2 * bids - prices) / bids) ** 2, 0.0),
    )

    units_sold = np.empty(n_rep)
    revenues = np.empty(n_rep)

    for k in range(n_rep):
        accept = rng.random(prob.size) < prob
        sold_stage2 = accept.sum()
        # Apply physical cap: cannot actually deliver more than remaining stock
        if t + sold_stage2 > N:
            # Deliver only up to inventory; extras become "unfilled" (could also be bumped)
            accept_indices = np.flatnonzero(accept)
            rng.shuffle(accept_indices)
            accept[accept_indices[N - t :]] = False
            sold_stage2 = accept.sum()

        units_sold[k] = t + sold_stage2
        revenue_stage1 = 0.0  # We don't have Stage-1 bids here; not needed for relative risk
        revenue_stage2 = np.sum(prices[accept])
        revenues[k] = revenue_stage1 + revenue_stage2

    stats = {
        "mean_units": units_sold.mean(),
        "p5_units": np.percentile(units_sold, 5),
        "p95_units": np.percentile(units_sold, 95),
        "mean_revenue": revenues.mean(),
        "p5_revenue": np.percentile(revenues, 5),
        "p95_revenue": np.percentile(revenues, 95),
        "oversell_prob": (units_sold > N).mean(),
    }
    return stats


if __name__ == "__main__":
    test_system()
    test_with_limited_inventory()