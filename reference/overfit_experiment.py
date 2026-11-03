"""과적합 사전 실험: 1-H-1 ReLU 신경망, Adam(lr=0.01), 시드 20개.
참 함수 sin(2.5x)+0.3x, 학습점 n, 잡음 sigma. 결과는 docs/SPEC.md 0절 참조.
실행: python3 overfit_experiment.py (NumPy 필요)
"""
import numpy as np
def f(x): return np.sin(2.5*x)+0.3*x
def run(H,epochs,seed,ntr=10,sig=0.15):
    r=np.random.default_rng(seed)
    xtr=np.sort(r.uniform(-1,1,ntr)); ytr=f(xtr)+r.normal(0,sig,ntr)
    xte=r.uniform(-0.95,0.95,40); yte=f(xte)+r.normal(0,sig,40)
    W1=r.normal(0,np.sqrt(2),H); b1=np.zeros(H); W2=r.normal(0,np.sqrt(2/H),H); b2=0.
    P=[W1,b1,W2,np.array([b2])]; m=[np.zeros_like(p) for p in P]; v=[np.zeros_like(p) for p in P]
    lr=0.01;B1,B2=0.9,0.999
    for t in range(1,epochs+1):
        z=np.outer(xtr,P[0])+P[1]; a=np.maximum(z,0); yh=a@P[2]+P[3][0]
        g=2*(yh-ytr)/ntr
        gW2=a.T@g; gb2=np.array([g.sum()]); da=np.outer(g,P[2])*(z>0)
        gW1=(da*xtr[:,None]).sum(0); gb1=da.sum(0)
        G=[gW1,gb1,gW2,gb2]
        for i in range(4):
            m[i]=B1*m[i]+(1-B1)*G[i]; v[i]=B2*v[i]+(1-B2)*G[i]**2
            P[i]=P[i]-lr*(m[i]/(1-B1**t))/(np.sqrt(v[i]/(1-B2**t))+1e-8)
    def pred(x): return np.maximum(np.outer(x,P[0])+P[1],0)@P[2]+P[3][0]
    return np.mean((pred(xtr)-ytr)**2), np.mean((pred(xte)-yte)**2)
for ep in [500,3000,20000]:
  for H in [1,2,4,8,16,64]:
    res=np.array([run(H,ep,s) for s in range(20)])
    print(ep,H,"train %.4f test %.4f (median test %.4f)"%(res[:,0].mean(),res[:,1].mean(),np.median(res[:,1])))
