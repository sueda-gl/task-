#!/usr/bin/env python3
"""Test the infeasible case directly"""
import numpy as np
from dynamic_pricing_system import stage2_pricing

# Create a scenario that should be infeasible:
# Large negative delta1, few customers with limited bids
P = 100.0
delta1 = -500.0  # Large deficit from Stage 1
inventory_left = 6
epsilon = 0.5 * P  # 50

# 6 customers with bids below P
remaining = [
    (1, 90.0),
    (2, 85.0),
    (3, 95.0),
    (4, 80.0),
    (5, 92.0),
    (6, 88.0)
]

print(f"Delta1: {delta1}")
print(f"Epsilon: {epsilon}")
print(f"Target adjustment needed: {-delta1 + epsilon} (must earn this much extra)")
print(f"Inventory left: {inventory_left}")
print(f"Remaining customers: {len(remaining)}")

# Maximum possible revenue increase
max_possible = sum(bid - P for _, bid in remaining)
print(f"Maximum possible adjustment (all at Bi): {max_possible}")
print(f"Can we meet target? {max_possible >= -delta1 + epsilon}")

# Run Stage 2
print("\nRunning Stage 2...")
prices, lambda_star = stage2_pricing(remaining, P, delta1, inventory_left)

print(f"\nLambda*: {lambda_star}")
print("Prices set:")
for (cid, price), (_, bid) in zip(prices, remaining):
    print(f"  Customer {cid}: bid={bid}, price={price:.2f}")

# Calculate actual adjustment
total_adj = 0.0
for (_, price), (_, bid) in zip(prices, remaining):
    if price <= bid:
        f_i = 1.0
    elif price <= 2 * bid:
        f_i = ((2 * bid - price) / bid) ** 2
    else:
        f_i = 0.0
    total_adj += (price - P) * f_i

print(f"\nActual adjustment achieved: {total_adj:.2f}")
print(f"Final balance: {delta1 + total_adj:.2f}")
print(f"Gap from target: {delta1 + total_adj - epsilon:.2f}") 