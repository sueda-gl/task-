import numpy as np
import dynamic_pricing_system as dps

def debug():
    P = 100.0
    N = 100
    np.random.seed(42)
    bids = [(i, np.random.uniform(0.51 * P, 1.5 * P)) for i in range(150)]
    accepted, remaining, t, delta1 = dps.stage1_allocation(bids, P, N)
    prices_stage2, lambda_star = dps.stage2_pricing(remaining, P, delta1)

    def acceptance_prob(P_i, B_i):
        if P_i <= B_i:
            return 1.0
        elif P_i <= 2 * B_i:
            return ((2 * B_i - P_i) / B_i) ** 2
        return 0.0

    total_adjustment = 0.0
    for (cid, P_i), (_, B_i) in zip(prices_stage2, remaining):
        total_adjustment += (P_i - P) * acceptance_prob(P_i, B_i)

    print(f"lambda_star = {lambda_star}")
    print(f"delta1      = {delta1}")
    print(f"Adjustment  = {total_adjustment}")
    print(f"Check       = {total_adjustment + delta1}")

    # Let's compute G directly using the internal logic of stage2_pricing
    # We replicate solve_single_customer_subproblem and compute_G here to test

    bids_only = np.array([bid for _, bid in remaining])

    def solve_single_customer_subproblem(B_i, lambda_val):
        candidates = []
        lambda_clipped = np.clip(lambda_val, -1e6, 1e6)
        min_price = 0.51 * P
        f_min = acceptance_prob(min_price, B_i)
        weight_min = 1 - lambda_clipped * (min_price - P)
        candidates.append((min_price, weight_min * f_min))
        f_B = 1.0
        weight_B = 1 - lambda_clipped * (B_i - P)
        candidates.append((B_i, weight_B * f_B))
        candidates.append((2 * B_i, 0.0))
        if abs(lambda_clipped) > 1e-9:
            P_star = (2 * (P + B_i) + 2 / lambda_clipped) / 3
            if B_i < P_star < 2 * B_i:
                f_star = acceptance_prob(P_star, B_i)
                weight_star = 1 - lambda_clipped * (P_star - P)
                candidates.append((P_star, weight_star * f_star))
        best_P_i, _ = max(candidates, key=lambda x: x[1])
        return best_P_i

    def compute_G(lambda_val):
        total = 0.0
        for B_i in bids_only:
            P_i_star = solve_single_customer_subproblem(B_i, lambda_val)
            f_i = acceptance_prob(P_i_star, B_i)
            total += (P_i_star - P) * f_i
        return total + delta1

    print(f"G(lambda_star) = {compute_G(lambda_star)}")
    for l in [-0.5, -0.2, -0.1, -0.05, -0.01, -5, -10]:
        print(f"G({l}) = {compute_G(l)}")

if __name__ == "__main__":
    debug() 