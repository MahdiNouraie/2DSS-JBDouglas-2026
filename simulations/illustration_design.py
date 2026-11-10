"""Design study for the slide 6 illustration.

Reports, for several correlation and signal levels, how often single Lasso fits
miss relevant variables, which determines how clearly the animation shows the
instability of one fit. Used to set the illustrative design (rho = 0.8, beta = 1.4 * lambda).
"""
import numpy as np
from sklearn.linear_model import Lasso
def run(rho, bmult, seed=1, B=100):
    rng=np.random.default_rng(seed); n,p=100,1000; n1=50; lam=np.sqrt(np.log(p)/n1)
    zS=rng.standard_normal((n,1)); XS=np.sqrt(rho)*zS+np.sqrt(1-rho)*rng.standard_normal((n,5))
    z=rng.standard_normal((n,1)); XO=np.sqrt(0.1)*z+np.sqrt(0.9)*rng.standard_normal((n,p-5))
    X=np.hstack([XS,XO]); X=(X-X.mean(0))/X.std(0)
    beta=np.zeros(p); beta[:5]=lam*bmult; y=X@beta+rng.standard_normal(n)
    f=np.zeros(p); sizes=[]; truecnt=[]
    for b in range(B):
        I=rng.choice(n,n1,replace=False); Xb=(X[I]-X[I].mean(0))/X[I].std(0)
        s=Lasso(alpha=lam,fit_intercept=False,max_iter=3000,tol=1e-4).fit(Xb,y[I]-y[I].mean()).coef_!=0
        f+=s; truecnt.append(s[:5].sum()); sizes.append(s.sum())
    f/=B; o=np.sort(f[5:])[::-1]
    return f[:5], o[:5], np.mean(truecnt), np.mean(sizes)
for rho in (0.5,0.7,0.8):
    for bm in ([2,2,2,2,2],[2.5,2.5,2.5,2.5,2.5],[3,3,3,3,3]):
        t,o,tc,sz=run(rho,np.array(bm,float))
        print(rho, bm[0], "relevant",np.round(t,2),"top others",np.round(o,2),"avg relevant/fit %.1f size %.1f"%(tc,sz))
