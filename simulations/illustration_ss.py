"""Illustrative design for the Stability Selection animation (slide 6).

Five correlated relevant variables (correlation 0.8, beta = 1.4 * lambda), so that
single Lasso fits visibly compete. This design is used for slide 6 only.
"""
import numpy as np, json
from sklearn.linear_model import Lasso
import os
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")
os.makedirs(OUT,exist_ok=True)
rng=np.random.default_rng(3); n,p=100,1000; n1=50; lam=np.sqrt(np.log(p)/n1); rho=0.8
zS=rng.standard_normal((n,1)); XS=np.sqrt(rho)*zS+np.sqrt(1-rho)*rng.standard_normal((n,5))
z=rng.standard_normal((n,1)); XO=np.sqrt(0.1)*z+np.sqrt(0.9)*rng.standard_normal((n,p-5))
X=np.hstack([XS,XO]); X=(X-X.mean(0))/X.std(0)
beta=np.zeros(p); beta[:5]=lam*1.4; y=X@beta+rng.standard_normal(n)
B=100; sel=[]; rows=[]
for b in range(B):
    I=rng.choice(n,n1,replace=False); Xb=(X[I]-X[I].mean(0))/X[I].std(0)
    s=Lasso(alpha=lam,fit_intercept=False,max_iter=3000,tol=1e-4).fit(Xb,y[I]-y[I].mean()).coef_!=0
    sel.append(s); rows.append(sorted(I.tolist()))
S=np.array(sel); f=S.mean(0)
top=5+np.argsort(-f[5:])[:5]
shown=list(range(5))+top.tolist()
recs=[(rows[b], [k for k,j in enumerate(shown) if S[b,j]]) for b in range(B)]
print("freq shown",np.round(f[shown],2)); print("first 3 resamples:",[r[1] for r in recs[:3]], "sizes", S[:3].sum(1))
json.dump({"recs":recs,"freq":f[shown].tolist()},open(os.path.join(OUT,"illustration_ss.json"),"w"))
