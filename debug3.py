import numpy as np, dynamic_pricing_system as dps
P=100; N=100
np.random.seed(789)
num=120
bids=[(i,float(np.random.uniform(0.8*P,1.25*P))) for i in range(num)]
acc,rem,t,delta = dps.stage1_allocation(bids,P,N)
print('Stage1 t',t,'delta',delta,'inventory',N-t)
prices, lam = dps.stage2_pricing(rem,P,delta,N-t)
print('lambda',lam)
# compute balance
adj=0; acc_prob=0
for (cid,Pi),(_,Bi) in zip(prices,rem):
    if Pi<=Bi: f=1.0
    elif Pi<=2*Bi: f=((2*Bi-Pi)/Bi)**2
    else: f=0.0
    adj+=(Pi-P)*f
    acc_prob+=f
print('adjust',adj,'balance',adj+delta,'expected_accept',acc_prob) 