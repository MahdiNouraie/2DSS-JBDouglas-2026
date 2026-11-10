"""Main simulation setting of the talk (motivating example of the paper, with 3 impostors).

Produces the selection frequencies for slides 8, 10 and 13 and the data for the
Lasso-path animation (slide 5). Output: output/sim2.json, output/sim2_data.npz.
"""
import numpy as np, json, time
from sklearn.linear_model import Lasso, lars_path
import os
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")
os.makedirs(OUT,exist_ok=True)
rng=np.random.default_rng(6)   # data set used throughout the talk
n,p=100,1000; n1=n//2; lam=np.sqrt(np.log(p)/n1)
K=3                                   # impostors
S=np.arange(5); C=np.arange(5,5+K); O=np.arange(5+K,p)
XS=rng.standard_normal((n,5)); a=np.array([0.4,0.4,0.4,0.4,-0.447])
XC=(XS@a)[:,None]+np.sqrt(1-a@a)*rng.standard_normal((n,K))
z=rng.standard_normal((n,1)); XO=np.sqrt(0.1)*z+np.sqrt(0.9)*rng.standard_normal((n,p-5-K))
X=np.hstack([XS,XC,XO]); X=(X-X.mean(0))/X.std(0)
beta=np.zeros(p); beta[:5]=lam*np.array([4,4,4,4,15])
y=X@beta+rng.standard_normal(n); y=y-y.mean()
print("corr impostors vs relevant:", np.round(np.corrcoef(X[:,:8].T)[5:8,:5],2))
def fit(Xb,yb):
    Xb=(Xb-Xb.mean(0))/Xb.std(0); yb=yb-yb.mean()
    return Lasso(alpha=lam,fit_intercept=False,max_iter=5000,tol=1e-4).fit(Xb,yb).coef_!=0
deltas=[0,0.025,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.75,0.9,1,1.25,1.5,2,2.5,3]
B=200; freq=np.zeros((len(deltas),p)); sel05=None; t0=time.time()
for di,d in enumerate(deltas):
    rec=[]
    for b in range(B):
        I=rng.choice(n,n1,replace=False)
        s=fit(X[I]+d*rng.standard_normal((n1,p)), y[I]); freq[di]+=s
        if d in (0,0.5) and b<100: rec.append((sorted(I.tolist()), np.flatnonzero(s[:8]).tolist(), int(s[8:].sum())))
    if d==0: sel0=rec
    if d==0.5: sel05=rec
    print(d,"relevant",np.round(freq[di,:5]/B,2),"imp",np.round(freq[di,C]/B,2),"oth %.3f"%(freq[di,O].mean()/B),"%.0fs"%(time.time()-t0),flush=True)
freq/=B
# Stability Selection without impostors (slide 6) and Lasso path (slide 5)
keep=np.r_[S,O]; Xn=X[:,keep]
show_oth=rng.choice(np.arange(5,len(keep)),25,replace=False)   # 25 irrelevant variables for the Lasso-path display
ssrec=[]; ssf=np.zeros(len(keep))
for b in range(B):
    I=rng.choice(n,n1,replace=False); s=fit(Xn[I],y[I]); ssf+=s
    if b<100: ssrec.append((sorted(I.tolist()), np.flatnonzero(s[:5]).tolist(), [int(k) for k in range(5) if False]+[int(np.flatnonzero(show_oth[:5]==j)[0]) for j in np.flatnonzero(s) if j in show_oth[:5]]))
ssf/=B
print("SS no impostors: relevant",np.round(ssf[:5],2),"5 shown others",np.round(ssf[show_oth[:5]],2),"other mean %.3f"%ssf[5:].mean())
json.dump({"deltas":deltas,"freq8":freq[:,:8].tolist(),"other_mean":freq[:,O].mean(1).tolist(),
           "sel0":sel0,"sel05":sel05,"ss_rec":ssrec,"ss_true":ssf[:5].tolist(),"ss_oth5":ssf[show_oth[:5]].tolist(),"ss_other_mean":float(ssf[5:].mean()),
           "lasso_cols":[0,1,2,3,4]+[int(keep[j]) for j in show_oth],"lam":lam},open(os.path.join(OUT,"main_setting.json"),"w"))
np.savez(os.path.join(OUT,"main_setting_data.npz"),X=X,y=y)
