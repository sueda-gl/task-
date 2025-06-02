P=100; B=120; lam=10

def acceptance(Pi,Bi):
    if Pi<=Bi:
        return 1.0
    elif Pi<=2*Bi:
        return ((2*Bi-Pi)/Bi)**2
    return 0.0
cands=[]
for Pi in [0.51*P, B, 2*B]:
    cands.append((Pi,(1+lam*(Pi-P))*acceptance(Pi,B)))
ps=2*B - (2 + 4*lam*B -2*lam*P)/(3*lam)
if B<ps<2*B:
    cands.append((ps,(1+lam*(ps-P))*acceptance(ps,B)))
print(cands)
print('best',max(cands,key=lambda x:x[1])) 