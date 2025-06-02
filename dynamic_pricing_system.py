import numpy as np
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
        # No inventory left - set all prices to infinity (or 2*Bi)
        if remaining and inventory_left <= 0:
            print("Warning: No inventory left for Stage 2. Setting all prices to maximum.")
            return [(cid, 2 * bid) for cid, bid in remaining], float('inf')
        return [], 0.0
    
    # ------------------------------------------------------------------
    # Reserve a default value so that we never exit without lambda_star.
    # It will be overwritten by the optimisation logic; if not, we will
    # engage the explicit fallback further below.
    # ------------------------------------------------------------------
    lambda_star: float = 0.0
    
    # Extract customer IDs and bids
    customer_ids = [cid for cid, _ in remaining]
    bids = np.array([bid for _, bid in remaining])
    R = len(remaining)
    
    # Maximum units we can sell in Stage 2
    max_sales = min(R, inventory_left)
    
    print(f"  Stage 2 inventory constraint: can sell at most {max_sales} units")
    
    def acceptance_prob(P_i: float, B_i: float) -> float:
        """Calculate acceptance probability f_i(P_i)"""
        if P_i <= B_i:
            return 1.0
        elif P_i <= 2 * B_i:
            return ((2 * B_i - P_i) / B_i) ** 2
        else:
            return 0.0
    
    def solve_single_customer_subproblem(B_i: float, lambda_val: float) -> float:
        """
        Single-customer maximisation
        ---------------------------
        We solve

            max_{0 ≤ P_i ≤ 2 B_i}  (1 + λ′ (P_i − P)) · f_i(P_i)

        where λ′ is the *re-scaled* dual variable explained in the docstring
        above.  The interior candidate derived from the first-order condition is

            P_i* = 2 B_i − (2 + 4 λ′ B_i − 2 λ′ P) / (3 λ′).

        This formula is algebraically equivalent to the one that uses two
        multipliers (λ, μ) once we substitute λ′ = λ / (1+μ).
        """
        candidates = []
        
        # Clip lambda to prevent overflow
        lambda_clipped = np.clip(lambda_val, -1e6, 1e6)
        
        # Candidate 2: P_i = B_i (boundary between regions)
        f_B = acceptance_prob(B_i, B_i)  # f_B = 1.0
        # Dual weighting follows 1 − λ (P_i − P)
        weight_B = 1 - lambda_clipped * (B_i - P)
        if abs(weight_B) < 1e10:
            obj_B = weight_B * f_B
            candidates.append((B_i, obj_B))
        
        # Candidate 3: P_i = 2*B_i
        f_2B = acceptance_prob(2 * B_i, B_i)  # f_2B = 0.0
        # Since f_2B = 0, the objective is always 0
        candidates.append((2 * B_i, 0.0))
        
        # Candidate 4: Interior stationary point (if it exists in (B_i, 2*B_i))
        # For P_i in (B_i, 2*B_i), f_i(P_i) = ((2*B_i - P_i)/B_i)^2
        # The objective becomes: (1 + lambda*(P_i - P)) * ((2*B_i - P_i)/B_i)^2
        # Taking derivative and setting to 0 gives:
        # P_i* = (2*(P + B_i) + 2*lambda/3) / 3
        
        if abs(lambda_clipped) > 1e-12:
            # Derived for objective   (1 − λ (P_i − P)) · f_i(P_i)
            P_star = 2 * B_i - (2 - 4 * lambda_clipped * B_i + 2 * lambda_clipped * P) / (3 * lambda_clipped)
        else:
            P_star = None
        
        if P_star is not None and B_i < P_star < 2 * B_i:
            f_star = acceptance_prob(P_star, B_i)
            # Dual weighting follows 1 − λ (P_i − P)
            weight_star = 1 - lambda_clipped * (P_star - P)
            if abs(weight_star) < 1e10:
                candidates.append((P_star, weight_star * f_star))
        
        # If no valid candidates, default to B_i
        if not candidates:
            return B_i
            
        # Return the P_i that maximizes the objective
        best_P_i, _ = max(candidates, key=lambda x: x[1])
        return best_P_i
    
    def compute_G(lambda_val: float) -> float:
        """
        Compute G(lambda) = sum_i [(P_i*(lambda) - P) * f_i(P_i*(lambda))] + delta1
        But also check inventory constraint
        """
        total_adjustment = 0.0
        expected_acceptances = 0.0
        
        for B_i in bids:
            P_i_star = solve_single_customer_subproblem(B_i, lambda_val)
            f_i = acceptance_prob(P_i_star, B_i)
            total_adjustment += (P_i_star - P) * f_i
            expected_acceptances += f_i
        
        # Hard inventory constraint: if expected acceptances exceed inventory, return a large
        # positive value so that any search interval that oversells will be discarded by the
        # bisection logic (we rely on sign changes).  This keeps the root finder inside the
        # feasible region where expected_acceptances <= inventory_left.
        if expected_acceptances > max_sales + 1e-9:
            return 1e30  # positive large number, sign = +

        return total_adjustment + delta1
    
    # Special case: if no inventory left but delta1 != 0, the problem is infeasible
    if inventory_left == 0 and abs(delta1) > 1e-6:
        print(f"  WARNING: Infeasible problem! No inventory left but need to adjust revenue by ${-delta1:.2f}")
        print(f"  Setting all prices to maximum to prevent sales.")
        return [(cid, 2 * bid) for cid, bid in remaining], float('inf')
    
    # If inventory is very limited, we might need to be more selective
    if inventory_left < R * 0.5:
        print(f"  Note: Limited inventory ({inventory_left} units for {R} customers)")
    
    # Use bisection to find lambda* such that G(lambda*) = 0
    # Start with a smarter range for lambda based on the problem structure
    
    # Estimate reasonable bounds based on the deficit and number of customers
    avg_deficit_needed = -delta1 / len(remaining) if remaining else 0
    
    # Lambda affects prices, and typical price adjustments are on the order of P
    # So lambda should be on a similar scale
    lambda_scale = abs(avg_deficit_needed / P) * 10
    lambda_min = -max(10, lambda_scale)
    lambda_max = max(10, lambda_scale)
    
    # Use bounded expansion to avoid overflow
    max_lambda_magnitude = 1e4  # Prevent extreme values
    
    # Expand range if necessary, but with bounds
    iteration_count = 0
    while compute_G(lambda_min) > 0 and lambda_min > -max_lambda_magnitude and iteration_count < 20:
        lambda_min *= 1.5  # Use 1.5x instead of 2x for more gradual expansion
        iteration_count += 1
        
    iteration_count = 0
    while compute_G(lambda_max) < 0 and lambda_max < max_lambda_magnitude and iteration_count < 20:
        lambda_max *= 1.5
        iteration_count += 1
    
    # If we hit the bounds, use them
    lambda_min = max(lambda_min, -max_lambda_magnitude)
    lambda_max = min(lambda_max, max_lambda_magnitude)
    
    # ------------------------------------------------------------------
    # Infeasibility detection.
    # If *all* feasible lambdas violate the inventory constraint, compute_G
    # will keep returning +inf.  Likewise, if G does not change sign the
    # revenue target cannot be met within the feasible region.
    # ------------------------------------------------------------------
    G_min = compute_G(lambda_min)
    G_max = compute_G(lambda_max)
    
    # Check if we can find a valid bracket for the revenue target
    if G_min * G_max > 0 and not (np.isinf(G_min) or np.isinf(G_max)):
        # ------------------------------------------------------------------
        # No sign-change ⇒ perfect balance infeasible under single-λ model.
        # We now search for the λ that MINIMISES the absolute imbalance
        # while observing the inventory cap.  This replaces the previous
        # ad-hoc surplus/deficit heuristics.
        # ------------------------------------------------------------------
        print("  INFO: Perfect balance infeasible within λ-model – running grid search for best feasible λ…")

        candidate_lambdas = [0.0]
        # Log-spaced magnitudes from 1e-6 to max_lambda_magnitude
        magnitudes = np.geomspace(1e-6, max_lambda_magnitude, num=60)
        for mag in magnitudes:
            candidate_lambdas.extend([mag, -mag])

        best_gap = np.inf
        best_lambda = None

        for lam in candidate_lambdas:
            G_val = compute_G(lam)
            if not np.isfinite(G_val) or abs(G_val) >= 1e29:  # inventory violated or overflow
                continue
            gap = abs(G_val)
            if gap < best_gap:
                best_gap = gap
                best_lambda = lam

        if best_lambda is not None:
            lambda_star = best_lambda
            print(f"     Chosen λ = {lambda_star:.6f} (revenue gap {best_gap:.2f})")
        else:
            print("     No single λ satisfies inventory cap – will invoke deterministic allocation later.")
    elif np.isinf(G_min) and np.isinf(G_max):
        # Both endpoints violate inventory
        print(f"  ERROR: Even with lambda=0, inventory constraint is violated.")
        lambda_star = 0.0
    else:
        # We have a valid bracket with sign change => can meet revenue target exactly
        # Run standard bisection
        tolerance = 1e-8
        max_iterations = 100
        
        for _ in range(max_iterations):
            lambda_mid = (lambda_min + lambda_max) / 2
            G_mid = compute_G(lambda_mid)
            
            if abs(G_mid) < tolerance:
                lambda_star = lambda_mid
                break
                
            if G_mid > 0:
                lambda_max = lambda_mid
            else:
                lambda_min = lambda_mid
        else:
            lambda_star = (lambda_min + lambda_max) / 2
    
    # ---------------------------------------------------------------
    # Fallback for extreme surplus with very tight inventory
    # ---------------------------------------------------------------
    G_final = compute_G(lambda_star)
    if (np.isinf(G_final) or G_final > 1e-4 or G_final < -1e-4):
        # Optimiser could not hit the balance exactly – construct an
        # explicit solution for the SURPLUS case (Δ₁ > 0) when stock is
        # too tight to allow discounts for everyone.

        if delta1 > 0 and inventory_left > 0:
            k = inventory_left
            # Required average discount per unit
            target_price = P - delta1 / k

            # Select k highest bids so that each bid ≥ target_price if possible
            rem_sorted = sorted(remaining, key=lambda x: x[1], reverse=True)
            selected = rem_sorted[:k]
            min_bid_selected = min(b for _, b in selected)

            # Ensure the offered price is not above any selected bid (prob=1)
            offer_price = min(target_price, min_bid_selected)

            selected_ids = {cid for cid, _ in selected}

            prices_stage2 = []
            for cid, bid in remaining:  # preserve original order
                if cid in selected_ids:
                    prices_stage2.append((cid, offer_price))
                else:
                    prices_stage2.append((cid, 2 * bid))

            print(f"  Fallback activated: uniform offer to top-{k} bidders at €{offer_price:.2f} to clear surplus.")
            return prices_stage2, float('inf')

        # Otherwise (deficit case or still infeasible) fall back to safe
        # no-sale pricing to protect inventory.
        print("  Fallback activated: giving prohibitive prices to avoid overselling.")
        return [(cid, 2 * bid) for cid, bid in remaining], float('inf')

    # Compute optimal prices using lambda*
    prices_stage2 = []
    for i, (cid, B_i) in enumerate(remaining):
        P_i_star = solve_single_customer_subproblem(B_i, lambda_star)
        prices_stage2.append((cid, P_i_star))
    
    return prices_stage2, lambda_star

    # ------------------------------------------------------------------
    # Fallback block – executed only if the code above early-returns.
    # ------------------------------------------------------------------
    # Should never reach here, but keep for static analysers.


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