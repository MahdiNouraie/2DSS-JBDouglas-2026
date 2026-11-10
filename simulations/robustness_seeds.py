"""Robustness check: repeats the main setting over seeds 1-12.

For each data set it reports the average selection frequencies of the relevant
variables and the impostors at delta = 0, 0.5 and 0.75, to confirm that the
pattern on the slides is not specific to the data set shown.
"""
import numpy as np
from sklearn.linear_model import Lasso
def run(seed,B=100,ds=(0,0.5,0.75)):
    rng=np.random.default_rng(seed); n,p=100,1000; n1=50; lam=np.sqrt(np.log(p)/n1); K=3
    XS=rng.standard_normal((n,5)); a=np.array([0.4,0.4,0.4,0.4,-0.447])
    XC=(XS@a)[:,None]+np.sqrt(1-a@a)*rng.standard_normal((n,K))
    z=rng.standard_normal((n,1)); XO=np.sqrt(0.1)*z+np.sqrt(0.9)*rng.standard_normal((n,p-5-K))
    X=np.hstack([XS,XC,XO]); X=(X-X.mean(0))/X.std(0)
    beta=np.zeros(p); beta[:5]=lam*np.array([4,4,4,4,15]); y=X@beta+rng.standard_normal(n)
    out=[]
    for d in ds:
        f=np.zeros(8)
        for b in range(B):
            I=rng.choice(n,n1,replace=False); Xb=X[I]+d*rng.standard_normal((n1,p)); Xb=(Xb-Xb.mean(0))/Xb.std(0)
            f+=(Lasso(alpha=lam,fit_intercept=False,max_iter=3000,tol=1e-4).fit(Xb,y[I]-y[I].mean()).coef_[:8]!=0)
        out.append(f/B)
    return out
for seed in range(1,13):
    o=run(seed)
    print(seed, "δ0 weak %.2f imp %.2f | δ.5 weak %.2f (min %.2f) imp %.2f (max %.2f) | δ.75 weak %.2f imp %.2f"%(
        o[0][:4].mean(),o[0][5:].mean(),o[1][:4].mean(),o[1][:4].min(),o[1][5:].mean(),o[1][5:].max(),o[2][:4].mean(),o[2][5:].mean()),flush=True)
