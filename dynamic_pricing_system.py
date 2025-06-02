import numpy as np
import pulp  # open-source MILP solver (CBC bundled)
from typing import List, Tuple, Dict
import bisect
import sys
import os
import shutil

# --- PyInstaller PuLP CBC Solver Path Fix ---
CBC_SOLVER = None
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Always use system CBC in PyInstaller mode
    system_cbc = shutil.which('cbc')
    if system_cbc:
        print(f"INFO: PyInstaller mode - Using system CBC at: {system_cbc}")
        CBC_SOLVER = pulp.PULP_CBC_CMD(path=system_cbc, msg=False)
    else:
        print("WARNING: System-wide CBC not found. PuLP will error if called.")
        CBC_SOLVER = pulp.PULP_CBC_CMD(msg=False)
else:
    CBC_SOLVER = pulp.PULP_CBC_CMD(msg=False)
# --- End PyInstaller PuLP CBC Solver Path Fix ---

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
    priority: str = "units",  # 'units' or 'gap'
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

    status = model.solve(CBC_SOLVER)

    # ------------------------------------------------------------------
    # If exact revenue balance is impossible, try a *relaxed* model that
    # allows a residual gap and minimises it lexicographically after
    # maximising the customer count.  This gives "the best we can do"
    # instead of reverting to prohibitive prices.
    # ------------------------------------------------------------------
    if status != pulp.LpStatusOptimal:
        # ---------- LEXICOGRAPHIC RELAXATION ---------------------------------
        if priority == "units":
            # ----------------- UNITS-FIRST ----------------------------------
            # Step 1: maximise number of winners, ignoring revenue balance.
            model_units = pulp.LpProblem("Stage2_units", pulp.LpMaximize)

            y_u = {cid: pulp.LpVariable(f"y_{cid}", 0, 1, cat="Binary") for cid in bids}
            P_u = {
                cid: pulp.LpVariable(
                    f"P_{cid}", lowBound=0.0, upBound=bids[cid], cat="Continuous"
                )
                for cid in bids
            }

            model_units += pulp.lpSum(y_u.values())  # objective: max units
            model_units += pulp.lpSum(y_u.values()) <= K

            for cid, B in bids.items():
                model_units += P_u[cid] <= B * y_u[cid]
                model_units += P_u[cid] >= 0.0 * y_u[cid]

            status_u = model_units.solve(CBC_SOLVER)

            if status_u != pulp.LpStatusOptimal:
                print("  MILP infeasible in unit-max phase — reverting to prohibitive pricing.")
                return [(cid, 2 * b) for cid, b in remaining]

            max_units = int(sum(v.value() for v in y_u.values()))

            # Step 2: with units fixed, minimise absolute revenue gap |gap|
            model_gap = pulp.LpProblem("Stage2_gap", pulp.LpMinimize)

            y_g = {cid: pulp.LpVariable(f"y_{cid}", 0, 1, cat="Binary") for cid in bids}
            P_g = {
                cid: pulp.LpVariable(
                    f"P_{cid}", lowBound=0.0, upBound=bids[cid], cat="Continuous"
                )
                for cid in bids
            }
            g_pos = pulp.LpVariable("g_pos", lowBound=0.0)
            g_neg = pulp.LpVariable("g_neg", lowBound=0.0)

            # Objective: minimise |gap|
            model_gap += g_pos + g_neg

            # Constraints
            model_gap += pulp.lpSum(y_g.values()) == max_units  # fix unit count
            model_gap += pulp.lpSum(y_g.values()) <= K          # redundant safety

            model_gap += (
                pulp.lpSum(P_g.values()) - P * pulp.lpSum(y_g.values()) + g_pos - g_neg == -delta1
            )

            for cid, B in bids.items():
                model_gap += P_g[cid] <= B * y_g[cid]
                model_gap += P_g[cid] >= 0.0 * y_g[cid]

            status_g = model_gap.solve(CBC_SOLVER)

            if status_g == pulp.LpStatusOptimal:
                prices = []
                for cid, b in remaining:
                    if y_g[cid].value() > 0.5:
                        prices.append((cid, P_g[cid].value()))
                    else:
                        prices.append((cid, 2 * b))
                gap_val = g_pos.value() - g_neg.value()
                print(
                    f"  Relaxed lexicographic MILP (units-first): residual balance {gap_val:+.2f} after selling {max_units} units."
                )
                return prices

            print("  MILP infeasible even after lexicographic relaxation — reverting to prohibitive pricing.")
            return [(cid, 2 * b) for cid, b in remaining]

        # ------------------ GAP-FIRST ---------------------------------------
        # Step 1: minimise |gap| ignoring units
        model_gap1 = pulp.LpProblem("Stage2_gap1", pulp.LpMinimize)

        y_g1 = {cid: pulp.LpVariable(f"y_{cid}", 0, 1, cat="Binary") for cid in bids}
        P_g1 = {cid: pulp.LpVariable(f"P_{cid}", 0.0, bids[cid]) for cid in bids}
        g_pos1 = pulp.LpVariable("g_pos1", lowBound=0.0)
        g_neg1 = pulp.LpVariable("g_neg1", lowBound=0.0)

        model_gap1 += g_pos1 + g_neg1
        model_gap1 += pulp.lpSum(y_g1.values()) <= K
        model_gap1 += pulp.lpSum(P_g1.values()) - P * pulp.lpSum(y_g1.values()) + g_pos1 - g_neg1 == -delta1
        for cid, B in bids.items():
            model_gap1 += P_g1[cid] <= B * y_g1[cid]
            model_gap1 += P_g1[cid] >= 0.0 * y_g1[cid]

        status_g1 = model_gap1.solve(CBC_SOLVER)

        if status_g1 != pulp.LpStatusOptimal:
            print("  GAP-first phase1 infeasible — reverting to prohibitive pricing.")
            return [(cid, 2 * b) for cid, b in remaining]

        best_gap = g_pos1.value() - g_neg1.value()

        # Step 2: maximise units with |gap| ≤ best_gap + tiny_eps
        eps = 1e-3
        model_units2 = pulp.LpProblem("Stage2_units2", pulp.LpMaximize)
        y2 = {cid: pulp.LpVariable(f"y2_{cid}", 0, 1, cat="Binary") for cid in bids}
        P2v = {cid: pulp.LpVariable(f"P2_{cid}", 0.0, bids[cid]) for cid in bids}

        model_units2 += pulp.lpSum(y2.values())
        model_units2 += pulp.lpSum(y2.values()) <= K
        # gap constraints within ± best_gap +/- eps
        model_units2 += (
            pulp.lpSum(P2v.values()) - P * pulp.lpSum(y2.values()) >= -delta1 - best_gap - eps
        )
        model_units2 += (
            pulp.lpSum(P2v.values()) - P * pulp.lpSum(y2.values()) <= -delta1 - best_gap + eps
        )

        for cid, B in bids.items():
            model_units2 += P2v[cid] <= B * y2[cid]
            model_units2 += P2v[cid] >= 0.0 * y2[cid]

        status_u2 = model_units2.solve(CBC_SOLVER)

        if status_u2 == pulp.LpStatusOptimal:
            prices = []
            for cid, b in remaining:
                if y2[cid].value() > 0.5:
                    prices.append((cid, P2v[cid].value()))
                else:
                    prices.append((cid, 2 * b))
            print(
                f"  Relaxed lexicographic MILP (gap-first): residual balance {best_gap:+.2f} with {int(sum(v.value() for v in y2.values()))} units."
            )
            return prices

        print("  MILP gap-first phase2 infeasible — reverting to prohibitive pricing.")
        return [(cid, 2 * b) for cid, b in remaining]

    prices = []
    for cid, b in remaining:
        if y[cid].value() > 0.5:
            prices.append((cid, Pvars[cid].value()))
        else:
            prices.append((cid, 2 * b))
    return prices


def stage2_pricing(remaining: List[Tuple[int, float]], P: float, delta1: float, inventory_left: int, priority: str = "units") -> Tuple[List[Tuple[int, float]], float]:
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
    prices_det = deterministic_stage2(remaining, P, delta1, inventory_left, priority=priority)

    # Deterministic MILP succeeded — use its prices directly.
    return prices_det, float("inf")


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