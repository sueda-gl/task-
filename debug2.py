import numpy as np, dynamic_pricing_system as dps
P=100.0; N=100
np.random.seed(123)
bids=[(i,np.random.uniform(0.51*P,1.2*P)) for i in range(80)]
acc,rem,t,delta1=dps.stage1_allocation(bids,P,N)
print('t',t,'delta1',delta1,'inventory_left',N-t)

def acceptance(Pi,Bi):
    if Pi<=Bi: return 1.0
    elif Pi<=2*Bi: return ((2*Bi-Pi)/Bi)**2
    return 0.0

def solve_single(Bi,l):
    lam=max(min(l,1e6),-1e6)
    min_price=0.51*P
    cands=[]
    f=acceptance(min_price,Bi)
    w=1-lam*(min_price-P)
    cands.append((min_price,w*f))
    f=1.0; w=1-lam*(Bi-P)
    cands.append((Bi,w*f))
    cands.append((2*Bi,0))
    if abs(lam)>1e-9:
        P_star=(2*(P+Bi)+2/lam)/3
        if Bi<P_star<2*Bi:
            f=acceptance(P_star,Bi)
            w=1-lam*(P_star-P)
            cands.append((P_star,w*f))
    return max(cands,key=lambda x:x[1])[0]

def compute_G(l):
    total=0.0; accept=0.0
    for _,Bi in rem:
        Pi=solve_single(Bi,l)
        f=acceptance(Pi,Bi)
        total+=(Pi-P)*f
        accept+=f
    return total+delta1,accept
for l in [-1000,-100,-50,-20,-10,-5,-2,-1,0,1,5,10,20,50,100,500,1000]:
    g,a=compute_G(l)
    print(f"l={l:6} g={g:8.2f} accept={a:5.2f}") 