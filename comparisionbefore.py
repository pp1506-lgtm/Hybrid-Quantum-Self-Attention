#Before MicroGPT optimization

# -*- coding: utf-8 -*-
"""
Three-Way Comparison: MicroGPT vs Classical MHA vs Hybrid Quantum Attention

Shared hyperparameters (all three models)
-----------------------------------------
  vocab_size  = 27   (26 letters + BOS)
  d_model     = 16
  n_head      = 4    (head_dim = 4)
  block_size  = 16
  n_layer     = 1
  n_qubits    = 4    (= log₂ 16 — Hybrid-Q only)
  train_steps = 100
"""

from __future__ import annotations
import math, random, time, os, warnings, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

warnings.filterwarnings("ignore")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ─────────────────────────────────────────────────────────────────
# Shared config
# ─────────────────────────────────────────────────────────────────
N_EMBD      = 16
BLOCK_SIZE  = 16
N_HEAD      = 4
HEAD_DIM    = N_EMBD // N_HEAD   # 4
N_LAYER     = 1
N_QUBITS    = 4
TRAIN_STEPS = 100
LR          = 0.01
T_1Q_NS     = 50.0
T_RO_NS     = 1_000.0

DARK="#0d1117"; PANEL="#161b22"; GRID="#21262d"; TEXT="#e6edf3"
C_MG="#79c0ff"; C_CL="#3fb950"; C_HQ="#d2a8ff"; ORNG="#f0883e"

# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
def load_names(path="names.txt"):
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        print("  Downloading names.txt …"); urllib.request.urlretrieve(url, path)
    docs = [l.strip() for l in open(path) if l.strip()]
    random.shuffle(docs); return docs

docs   = load_names()
CHARS  = sorted(set("".join(docs)))   # ['a'..'z']
BOS    = len(CHARS)                   # 26
VOCAB  = BOS + 1                      # 27
CH2I   = {c:i for i,c in enumerate(CHARS)}
encode = lambda doc: [BOS] + [CH2I[c] for c in doc] + [BOS]

split    = int(0.9*len(docs))
TR_DOCS  = docs[:split]
VAL_DOCS = docs[split:]
print(f"  {len(docs):,} names  |  train={len(TR_DOCS):,}  val={len(VAL_DOCS):,}  vocab={VOCAB}")

def make_batch(doc_list, max_docs=32):
    seqs = [encode(d) for d in doc_list[:max_docs]]
    maxL = min(BLOCK_SIZE+1, max(len(s) for s in seqs))
    ids  = torch.full((len(seqs), maxL), BOS, dtype=torch.long)
    for i,s in enumerate(seqs):
        t=s[:maxL]; ids[i,:len(t)]=torch.tensor(t)
    return ids

# ─────────────────────────────────────────────────────────────────
# 1. MicroGPT — Karpathy's pure-Python autograd, verbatim logic
# ─────────────────────────────────────────────────────────────────
class Value:
    __slots__=("data","grad","_children","_local_grads")
    def __init__(self,data,children=(),local_grads=()):
        self.data=data;self.grad=0;self._children=children;self._local_grads=local_grads
    def __add__(self,o):
        o=o if isinstance(o,Value) else Value(o)
        return Value(self.data+o.data,(self,o),(1,1))
    def __mul__(self,o):
        o=o if isinstance(o,Value) else Value(o)
        return Value(self.data*o.data,(self,o),(o.data,self.data))
    def __pow__(self,o): return Value(self.data**o,(self,),(o*self.data**(o-1),))
    def log(self):  return Value(math.log(self.data),(self,),(1/self.data,))
    def exp(self):  return Value(math.exp(self.data),(self,),(math.exp(self.data),))
    def relu(self): return Value(max(0,self.data),(self,),(float(self.data>0),))
    def __neg__(self): return self*-1
    def __radd__(self,o): return self+o
    def __sub__(self,o):  return self+(-o)
    def __rmul__(self,o): return self*o
    def __truediv__(self,o):  return self*o**-1
    def backward(self):
        topo=[]; visited=set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self); self.grad=1
        for v in reversed(topo):
            for child,lg in zip(v._children,v._local_grads):
                child.grad+=lg*v.grad

class MicroGPT:
    def __init__(self):
        mat=lambda r,c,s=0.08:[[Value(random.gauss(0,s)) for _ in range(c)] for _ in range(r)]
        self.sd={"wte":mat(VOCAB,N_EMBD),"wpe":mat(BLOCK_SIZE,N_EMBD),"lm_head":mat(VOCAB,N_EMBD)}
        for i in range(N_LAYER):
            for k in ["attn_wq","attn_wk","attn_wv","attn_wo"]:
                self.sd[f"layer{i}.{k}"]=mat(N_EMBD,N_EMBD)
            self.sd[f"layer{i}.mlp_fc1"]=mat(4*N_EMBD,N_EMBD)
            self.sd[f"layer{i}.mlp_fc2"]=mat(N_EMBD,4*N_EMBD)
        self.params=[p for m in self.sd.values() for row in m for p in row]
        self.m=[0.]*len(self.params); self.v=[0.]*len(self.params)

    @staticmethod
    def _lin(x,w): return [sum(wi*xi for wi,xi in zip(wo,x)) for wo in w]
    @staticmethod
    def _smx(lg):
        mx=max(v.data for v in lg); ex=[(v-mx).exp() for v in lg]; s=sum(ex)
        return [e/s for e in ex]
    @staticmethod
    def _rms(x):
        ms=sum(xi*xi for xi in x)/len(x); sc=(ms+1e-5)**-0.5; return [xi*sc for xi in x]

    def fwd(self,tok,pos,keys,vals):
        x=[t+p for t,p in zip(self.sd["wte"][tok],self.sd["wpe"][pos])]
        x=self._rms(x)
        for li in range(N_LAYER):
            xr=x; x=self._rms(x)
            q=self._lin(x,self.sd[f"layer{li}.attn_wq"])
            k=self._lin(x,self.sd[f"layer{li}.attn_wk"])
            v=self._lin(x,self.sd[f"layer{li}.attn_wv"])
            keys[li].append(k); vals[li].append(v)
            xa=[]
            for h in range(N_HEAD):
                hs=h*HEAD_DIM
                qh=q[hs:hs+HEAD_DIM]
                kh=[ki[hs:hs+HEAD_DIM] for ki in keys[li]]
                vh=[vi[hs:hs+HEAD_DIM] for vi in vals[li]]
                al=[sum(qh[j]*kh[t][j] for j in range(HEAD_DIM))/HEAD_DIM**0.5 for t in range(len(kh))]
                aw=self._smx(al)
                xa.extend([sum(aw[t]*vh[t][j] for t in range(len(vh))) for j in range(HEAD_DIM)])
            x=self._lin(xa,self.sd[f"layer{li}.attn_wo"]); x=[a+b for a,b in zip(x,xr)]
            xr=x; x=self._rms(x); x=self._lin(x,self.sd[f"layer{li}.mlp_fc1"])
            x=[xi.relu() for xi in x]; x=self._lin(x,self.sd[f"layer{li}.mlp_fc2"])
            x=[a+b for a,b in zip(x,xr)]
        return self._lin(x,self.sd["lm_head"])

    def train_step(self,doc,step):
        toks=encode(doc); n=min(BLOCK_SIZE,len(toks)-1)
        keys=[[] for _ in range(N_LAYER)]; vals=[[] for _ in range(N_LAYER)]
        lss=[]
        for pos in range(n):
            lg=self.fwd(toks[pos],pos,keys,vals); pr=self._smx(lg)
            lss.append(-pr[toks[pos+1]].log())
        loss=(1/n)*sum(lss); loss.backward()
        lrt=LR*(1-step/TRAIN_STEPS); b1,b2,ep=0.85,0.99,1e-8
        for i,p in enumerate(self.params):
            self.m[i]=b1*self.m[i]+(1-b1)*p.grad
            self.v[i]=b2*self.v[i]+(1-b2)*p.grad**2
            mh=self.m[i]/(1-b1**(step+1)); vh=self.v[i]/(1-b2**(step+1))
            p.data-=lrt*mh/(vh**0.5+ep); p.grad=0
        return loss.data

    def val_ppl(self,docs,cap=150):
        tot,n=0.,0
        for doc in docs[:cap]:
            toks=encode(doc); nt=min(BLOCK_SIZE,len(toks)-1)
            keys=[[] for _ in range(N_LAYER)]; vals=[[] for _ in range(N_LAYER)]
            for pos in range(nt):
                lg=self.fwd(toks[pos],pos,keys,vals); pr=self._smx(lg)
                tot+=math.log(max(pr[toks[pos+1]].data,1e-9)); n+=1
        return math.exp(-tot/n) if n else float("inf")

    def generate(self,n=8,temp=0.8):
        out=[]
        for _ in range(n):
            keys=[[] for _ in range(N_LAYER)]; vals=[[] for _ in range(N_LAYER)]
            tok=BOS; s=[]
            for pos in range(BLOCK_SIZE):
                lg=self.fwd(tok,pos,keys,vals); pr=self._smx([l/temp for l in lg])
                tok=random.choices(range(VOCAB),weights=[p.data for p in pr])[0]
                if tok==BOS: break
                s.append(CHARS[tok])
            out.append("".join(s))
        return out

    def heat(self,doc):
        """Return (L,L) attention weight matrix for head-0 over a doc."""
        toks=encode(doc); n=min(BLOCK_SIZE,len(toks)-1)
        keys=[[] for _ in range(N_LAYER)]; vals=[[] for _ in range(N_LAYER)]
        rows=[]
        for pos in range(n):
            x=[t+p for t,p in zip(self.sd["wte"][toks[pos]],self.sd["wpe"][pos])]
            x=self._rms(x)
            q=self._lin(x,self.sd["layer0.attn_wq"])
            k=self._lin(x,self.sd["layer0.attn_wk"])
            v=self._lin(x,self.sd["layer0.attn_wv"])
            keys[0].append(k); vals[0].append(v)
            qh=q[:HEAD_DIM]; kh=[ki[:HEAD_DIM] for ki in keys[0]]
            al=[sum(qh[j]*kh[t][j] for j in range(HEAD_DIM))/HEAD_DIM**0.5 for t in range(len(kh))]
            aw=self._smx(al); rows.append([p.data for p in aw])
        L=n; mat=np.zeros((L,L))
        for i,row in enumerate(rows): mat[i,:len(row)]=row
        return mat, toks[:L]


# ─────────────────────────────────────────────────────────────────
# 2. Classical MultiHeadAttention (PyTorch, mirroring attention.py)
# ─────────────────────────────────────────────────────────────────
class ClassicalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb =nn.Embedding(VOCAB,N_EMBD)
        self.pos =nn.Embedding(BLOCK_SIZE,N_EMBD)
        self.Wq  =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.Wk  =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.Wv  =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.Wo  =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.n1  =nn.LayerNorm(N_EMBD)
        self.ff  =nn.Sequential(nn.Linear(N_EMBD,4*N_EMBD,bias=False),
                                 nn.ReLU(),
                                 nn.Linear(4*N_EMBD,N_EMBD,bias=False))
        self.n2  =nn.LayerNorm(N_EMBD)
        self.head=nn.Linear(N_EMBD,VOCAB,bias=False)
        self.last_score_ms=0.

    def forward(self,ids):
        B,L=ids.shape
        x=self.emb(ids)+self.pos(torch.arange(L))
        # MHA
        Q=self.Wq(x).view(B,L,N_HEAD,HEAD_DIM).transpose(1,2)
        K=self.Wk(x).view(B,L,N_HEAD,HEAD_DIM).transpose(1,2)
        V=self.Wv(x).view(B,L,N_HEAD,HEAD_DIM).transpose(1,2)
        t0=time.perf_counter()
        S=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(HEAD_DIM)
        self.last_score_ms=(time.perf_counter()-t0)*1e3
        W=F.softmax(S,dim=-1)
        out=torch.matmul(W,V).transpose(1,2).contiguous().view(B,L,N_EMBD)
        x=self.n1(x+self.Wo(out))
        x=self.n2(x+self.ff(x))
        return self.head(x),W   # (B,L,V),(B,H,L,L)

    def generate(self,n=8,temp=0.8):
        self.eval(); out=[]
        with torch.no_grad():
            for _ in range(n):
                ctx=[BOS]; s=[]
                for _ in range(BLOCK_SIZE):
                    ids=torch.tensor([ctx[-BLOCK_SIZE:]]).long()
                    lg,_=self(ids); lg=lg[0,-1]/temp
                    tok=torch.multinomial(F.softmax(lg,dim=-1),1).item()
                    if tok==BOS: break
                    ctx.append(tok); s.append(CHARS[tok])
                out.append("".join(s))
        return out

    def val_ppl(self,docs,cap=300):
        self.eval(); tot,n=0.,0
        with torch.no_grad():
            for doc in docs[:cap]:
                ids=make_batch([doc])
                if ids.shape[1]<2: continue
                inp,tgt=ids[:,:-1],ids[:,1:]
                lg,_=self(inp)
                mask=(tgt!=BOS).reshape(-1)
                if not mask.any(): continue
                tot+=F.cross_entropy(lg.reshape(-1,VOCAB)[mask],tgt.reshape(-1)[mask],reduction="sum").item()
                n+=mask.sum().item()
        self.train()
        return math.exp(tot/n) if n else float("inf")

    def heat(self,doc):
        self.eval()
        with torch.no_grad():
            ids=make_batch([doc]); inp=ids[:,:-1]
            _,W=self(inp)
        self.train()
        return W[0,0].numpy(), encode(doc)[:-1][:inp.shape[1]]


def train_torch(model,label,steps=TRAIN_STEPS):
    opt=torch.optim.Adam(model.parameters(),lr=LR,betas=(0.85,0.99))
    sched=torch.optim.lr_scheduler.LinearLR(opt,1.,0.01,total_iters=steps)
    losses=[]; val_ppls=[]
    model.train()
    for step in range(steps):
        s=step%max(1,len(TR_DOCS)-32)
        ids=make_batch(TR_DOCS[s:s+32])
        inp,tgt=ids[:,:-1],ids[:,1:]
        if inp.shape[1]==0: continue
        lg,_=model(inp)
        loss=F.cross_entropy(lg.reshape(-1,VOCAB),tgt.reshape(-1),ignore_index=BOS)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if (step+1)%25==0:
            vp=model.val_ppl(VAL_DOCS)
            val_ppls.append(vp)
            print(f"    [{label}] step {step+1:3d}/{steps}  loss={loss.item():.4f}  val_ppl={vp:.2f}")
    return losses, val_ppls


# ─────────────────────────────────────────────────────────────────
# 3. Hybrid Quantum LM  (Qiskit angle encoding + PyTorch)
# ─────────────────────────────────────────────────────────────────
_SIM = AerSimulator(method="statevector")

def _qenc_warm():
    qc=QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS): qc.ry(0.,i)
    qc.save_statevector()
    tc=transpile(qc,_SIM,optimization_level=1)
    _SIM.run(tc).result()
    return tc

_TEMPLATE = _qenc_warm()

def encode_batch_q(vectors: np.ndarray) -> np.ndarray:
    """Batch angle-encode (B,d) → (B,2^n) statevectors. One Qiskit job."""
    B=vectors.shape[0]
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    norms=np.where(norms<1e-8,1.,norms)
    circuits=[]
    for i in range(B):
        ang=math.pi*vectors[i,:N_QUBITS]/norms[i,0]
        qc=QuantumCircuit(N_QUBITS)
        for j in range(N_QUBITS): qc.ry(float(ang[j]),j)
        qc.save_statevector(); circuits.append(qc)
    tcs=transpile(circuits,_SIM,optimization_level=1)
    res=_SIM.run(tcs).result()
    return np.stack([np.array(res.get_statevector(i)) for i in range(B)])

def _proj_hw_ms(n_circuits):
    return n_circuits*(N_QUBITS*T_1Q_NS+T_RO_NS)/1e6


class HybridLM(nn.Module):
    """Q/K/V projections classical; Q·Kᵀ via batched quantum statevector overlaps."""
    def __init__(self):
        super().__init__()
        self.emb  =nn.Embedding(VOCAB,N_EMBD)
        self.pos  =nn.Embedding(BLOCK_SIZE,N_EMBD)
        self.Wq   =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.Wk   =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.Wv   =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.Wo   =nn.Linear(N_EMBD,N_EMBD,bias=False)
        self.log_t=nn.Parameter(torch.zeros(1))
        self.n1   =nn.LayerNorm(N_EMBD)
        self.ff   =nn.Sequential(nn.Linear(N_EMBD,4*N_EMBD,bias=False),
                                  nn.ReLU(),
                                  nn.Linear(4*N_EMBD,N_EMBD,bias=False))
        self.n2   =nn.LayerNorm(N_EMBD)
        self.head  =nn.Linear(N_EMBD,VOCAB,bias=False)
        self.last_score_ms=0.; self.last_hw_ms=0.

    def _q_scores(self, Q: torch.Tensor, K: torch.Tensor):
        """(B,L,d) → (B,L,L) score tensor via ONE batched quantum job."""
        B,L,_=Q.shape
        Qnp=Q.detach().numpy().reshape(B*L,-1)
        Knp=K.detach().numpy().reshape(B*L,-1)
        t0=time.perf_counter()
        all_sv=encode_batch_q(np.vstack([Qnp,Knp]))   # (2BL, 2^n)
        Qsv=all_sv[:B*L].reshape(B,L,-1)
        Ksv=all_sv[B*L:].reshape(B,L,-1)
        S=np.stack([np.real(Qsv[b]@Ksv[b].conj().T) for b in range(B)])
        self.last_score_ms=(time.perf_counter()-t0)*1e3
        self.last_hw_ms=_proj_hw_ms(2*B*L)
        return torch.tensor(S,dtype=torch.float32)*torch.exp(self.log_t)

    def forward(self,ids):
        B,L=ids.shape
        x=self.emb(ids)+self.pos(torch.arange(L))
        Q=self.Wq(x); K=self.Wk(x); V=self.Wv(x)
        S=self._q_scores(Q,K)        # (B,L,L)
        W=F.softmax(S,dim=-1)
        ctx=torch.bmm(W,V)
        x=self.n1(x+self.Wo(ctx))
        x=self.n2(x+self.ff(x))
        return self.head(x),W.unsqueeze(1)  # (B,L,V),(B,1,L,L)

    def generate(self,n=8,temp=0.8):
        self.eval(); out=[]
        with torch.no_grad():
            for _ in range(n):
                ctx=[BOS]; s=[]
                for _ in range(BLOCK_SIZE):
                    ids=torch.tensor([ctx[-BLOCK_SIZE:]]).long()
                    lg,_=self(ids); lg=lg[0,-1]/temp
                    tok=torch.multinomial(F.softmax(lg,dim=-1),1).item()
                    if tok==BOS: break
                    ctx.append(tok); s.append(CHARS[tok])
                out.append("".join(s))
        return out

    def val_ppl(self,docs,cap=80):
        self.eval(); tot,n=0.,0
        with torch.no_grad():
            for doc in docs[:cap]:
                ids=make_batch([doc])
                if ids.shape[1]<2: continue
                inp,tgt=ids[:,:-1],ids[:,1:]
                lg,_=self(inp)
                mask=(tgt!=BOS).reshape(-1)
                if not mask.any(): continue
                tot+=F.cross_entropy(lg.reshape(-1,VOCAB)[mask],tgt.reshape(-1)[mask],reduction="sum").item()
                n+=mask.sum().item()
        self.train()
        return math.exp(tot/n) if n else float("inf")

    def heat(self,doc):
        self.eval()
        with torch.no_grad():
            ids=make_batch([doc]); inp=ids[:,:-1]
            _,W=self(inp)
        self.train()
        return W[0,0].numpy(), encode(doc)[:-1][:inp.shape[1]]


def train_hybrid(model,steps=TRAIN_STEPS):
    opt=torch.optim.Adam(model.parameters(),lr=LR,betas=(0.85,0.99))
    sched=torch.optim.lr_scheduler.LinearLR(opt,1.,0.01,total_iters=steps)
    losses=[]; val_ppls=[]
    model.train()
    for step in range(steps):
        s=step%max(1,len(TR_DOCS)-8)
        ids=make_batch(TR_DOCS[s:s+8],max_docs=8)   # smaller batch = fewer qubits
        inp,tgt=ids[:,:-1],ids[:,1:]
        if inp.shape[1]==0: continue
        lg,_=model(inp)
        loss=F.cross_entropy(lg.reshape(-1,VOCAB),tgt.reshape(-1),ignore_index=BOS)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
        if (step+1)%25==0:
            vp=model.val_ppl(VAL_DOCS)
            val_ppls.append(vp)
            print(f"    [hybrid-Q ] step {step+1:3d}/{steps}  loss={loss.item():.4f}  val_ppl={vp:.2f}")
    return losses, val_ppls


# ─────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────
def benchmark_qkt(seq_lens=(8,12,16),n_rep=4):
    """Compare Q·Kᵀ wall-clock for all three implementations."""
    print("\n  Benchmarking Q·Kᵀ score computation …")
    res={k:[] for k in ["seq_len","microgpt_ms","classical_ms","qsim_ms","proj_hw_ms"]}
    for L in seq_lens:
        # MicroGPT: Value scalars
        Qv=[[Value(float(x)) for x in r] for r in np.random.randn(L,HEAD_DIM)]
        Kv=[[Value(float(x)) for x in r] for r in np.random.randn(L,HEAD_DIM)]
        t_mg=[]
        for _ in range(n_rep):
            t0=time.perf_counter()
            [[sum(Qv[i][j]*Kv[t][j] for j in range(HEAD_DIM))/HEAD_DIM**0.5
              for t in range(L)] for i in range(L)]
            t_mg.append((time.perf_counter()-t0)*1e3)
        t_mg=float(np.median(t_mg))

        # Classical PyTorch
        Qt=torch.randn(1,N_HEAD,L,HEAD_DIM); Kt=torch.randn(1,N_HEAD,L,HEAD_DIM)
        t_cl=[]
        for _ in range(n_rep):
            t0=time.perf_counter()
            torch.matmul(Qt,Kt.transpose(-2,-1))/math.sqrt(HEAD_DIM)
            t_cl.append((time.perf_counter()-t0)*1e3)
        t_cl=float(np.median(t_cl))

        # Quantum sim
        Qn=np.random.randn(L,N_EMBD).astype(np.float32)
        Kn=np.random.randn(L,N_EMBD).astype(np.float32)
        arr=np.vstack([Qn,Kn])
        encode_batch_q(arr)  # warm-up
        t_qs=[]
        for _ in range(max(1,n_rep-1)):
            t0=time.perf_counter()
            sv=encode_batch_q(arr); np.real(sv[:L]@sv[L:].conj().T)
            t_qs.append((time.perf_counter()-t0)*1e3)
        t_qs=float(np.median(t_qs))
        t_hw=_proj_hw_ms(2*L)

        res["seq_len"].append(L)
        res["microgpt_ms"].append(t_mg)
        res["classical_ms"].append(t_cl)
        res["qsim_ms"].append(t_qs)
        res["proj_hw_ms"].append(t_hw)
        print(f"    L={L}: MicroGPT={t_mg:.2f}ms  Classical={t_cl:.4f}ms  "
              f"Qiskit-sim={t_qs:.1f}ms  QHW-proj={t_hw:.4f}ms")
    return res


def crossover_analysis():
    d_vals=[64,128,256,512,1024,2048,4096]; SEQ=16
    tc_list=[]; tp_list=[]
    for d in d_vals:
        nq=math.ceil(math.log2(d))
        Q=np.random.randn(SEQ,d).astype(np.float32); K=np.random.randn(SEQ,d).astype(np.float32)
        ts=[]
        for _ in range(10): t0=time.perf_counter(); Q@K.T; ts.append(time.perf_counter()-t0)
        tc_list.append(float(np.median(ts))*1e3)
        tp_list.append(2*SEQ*(nq*T_1Q_NS+T_RO_NS)/1e6)
    return d_vals, tc_list, tp_list


# ─────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────
def _ax(ax,title):
    ax.set_facecolor(PANEL); ax.tick_params(colors=TEXT,labelsize=8)
    for l in (ax.xaxis.label,ax.yaxis.label,ax.title): l.set_color(TEXT)
    ax.set_title(title,fontsize=9.5,fontweight="bold",pad=7)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(color=GRID,ls="--",lw=0.5)


def make_figure(mg_l,cl_l,hq_l, mg_vp,cl_vp,hq_vp,
                bench,cross,
                mg_h,cl_h,hq_h,heat_toks,
                mg_s,cl_s,hq_s,ppls,path):
    fig=plt.figure(figsize=(20,14),facecolor=DARK)
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.52,wspace=0.38,
                         left=0.06,right=0.97,top=0.93,bottom=0.06)

    # (0,0) Training loss
    ax=fig.add_subplot(gs[0,0]); _ax(ax,"Training Loss  (names.txt · char-level)")
    ax.plot(mg_l,color=C_MG,lw=1.4,alpha=.9,label="MicroGPT  (pure-Python)")
    ax.plot(cl_l,color=C_CL,lw=1.4,alpha=.9,label="Classical MHA  (PyTorch)")
    ax.plot(hq_l,color=C_HQ,lw=1.4,alpha=.9,label="Hybrid-Q  (Qiskit+PyTorch)")
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss")
    ax.legend(fontsize=7.5,facecolor=PANEL,labelcolor=TEXT,edgecolor=GRID)

    # (0,1) Q·Kᵀ benchmark
    ax=fig.add_subplot(gs[0,1]); _ax(ax,"Q·Kᵀ Score Computation Time")
    sl=bench["seq_len"]; x=np.arange(len(sl)); w=0.22
    ax.bar(x-w,  bench["microgpt_ms"], w,color=C_MG,alpha=.9,label="MicroGPT")
    ax.bar(x,    bench["classical_ms"],w,color=C_CL,alpha=.9,label="Classical")
    ax.bar(x+w,  bench["proj_hw_ms"], w,color=C_HQ,alpha=.9,label="Quantum HW (projected)")
    ax.set_xticks(x); ax.set_xticklabels([f"L={s}" for s in sl])
    ax.set_ylabel("Time (ms)"); ax.set_xlabel("Sequence length")
    ax.legend(fontsize=7.5,facecolor=PANEL,labelcolor=TEXT,edgecolor=GRID)
    for i,(tm,tq) in enumerate(zip(bench["microgpt_ms"],bench["proj_hw_ms"])):
        spd=tm/tq if tq>0 else 0
        ax.annotate(f"{spd:.0f}×",xy=(i+w,tq),xytext=(0,4),textcoords="offset points",
                    ha="center",fontsize=7,color=C_HQ,fontweight="bold")

    # (0,2) Crossover
    ax=fig.add_subplot(gs[0,2]); _ax(ax,"Crossover: d_model vs Q·Kᵀ  (seq=16)")
    d_arr,tc_arr,tp_arr=cross
    ax.plot(d_arr,tc_arr,"o-", color=C_CL,lw=2,ms=5,label="Classical  O(L²·d)")
    ax.plot(d_arr,tp_arr,"s--",color=C_HQ,lw=2,ms=5,label="Quantum HW  O(log d)")
    ax.set_xlabel("d_model"); ax.set_ylabel("Time (ms)"); ax.set_xscale("log",base=2)
    ax.legend(fontsize=8,facecolor=PANEL,labelcolor=TEXT,edgecolor=GRID)
    for d_,tc_,tp_ in zip(d_arr,tc_arr,tp_arr):
        if tp_<=tc_:
            ax.axvline(d_,color=C_HQ,ls=":",lw=1.5,alpha=.7)
            ax.annotate(f"crossover\nd={d_:,}",xy=(d_,(tc_+tp_)/2),fontsize=7,color=C_HQ,
                        xytext=(6,0),textcoords="offset points"); break
    d_np=np.array(d_arr); tc_np=np.array(tc_arr); tp_np=np.array(tp_arr)
    m2=tp_np<tc_np
    if m2.any(): ax.fill_between(d_np[m2],tc_np[m2],tp_np[m2],alpha=.1,color=C_HQ)

    # Row 1: attention heatmaps
    probe_lbl=[CHARS[t] if t<BOS else "●" for t in heat_toks]
    for col,(heat,cmap,title) in enumerate(zip(
            [mg_h,cl_h,hq_h],
            ["Blues","Greens","Purples"],
            ["MicroGPT  (head 0)","Classical MHA  (head 0)","Hybrid-Q  (statevector)"])):
        ax=fig.add_subplot(gs[1,col]); ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color(GRID)
        L=heat.shape[0]
        im=ax.imshow(heat,cmap=cmap,vmin=0,vmax=max(heat.max(),1e-6))
        ax.set_title(f"Attention Heatmap · {title}",fontsize=9,fontweight="bold",color=TEXT,pad=6)
        ax.set_xticks(range(L)); ax.set_xticklabels(probe_lbl[:L],rotation=45,fontsize=9,color=TEXT)
        ax.set_yticks(range(L)); ax.set_yticklabels(probe_lbl[:L],fontsize=9,color=TEXT)
        ax.set_xlabel("Key tokens",color=TEXT); ax.set_ylabel("Query tokens",color=TEXT)
        cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.ax.tick_params(labelcolor=TEXT)

    # (2,0) Val PPL curves
    ax=fig.add_subplot(gs[2,0]); _ax(ax,"Validation Perplexity  (every 25 steps)")
    ckpts=[25,50,75,100]
    for vp,col,lbl in [(mg_vp,C_MG,"MicroGPT"),(cl_vp,C_CL,"Classical"),(hq_vp,C_HQ,"Hybrid-Q")]:
        if vp: ax.plot(ckpts[:len(vp)],vp,"o-",color=col,lw=1.5,ms=5,label=lbl)
    ax.set_xlabel("Training step"); ax.set_ylabel("Perplexity")
    ax.legend(fontsize=8,facecolor=PANEL,labelcolor=TEXT,edgecolor=GRID)

    # (2,1) Generated samples text panel
    ax=fig.add_subplot(gs[2,1]); ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Generated Names  (temperature = 0.8)",fontsize=9.5,fontweight="bold",color=TEXT,pad=7)
    y=0.95
    for lbl,samps,col in [("MicroGPT",mg_s,C_MG),("Classical",cl_s,C_CL),("Hybrid-Q",hq_s,C_HQ)]:
        ax.text(0.03,y,f"{lbl}:",color=col,fontsize=9,fontweight="bold",
                transform=ax.transAxes,va="top"); y-=0.05
        for s in samps[:6]:
            ax.text(0.10,y,s,color=TEXT,fontsize=8.5,fontfamily="monospace",
                    transform=ax.transAxes,va="top"); y-=0.05
        y-=0.04

    # (2,2) Final PPL bar
    ax=fig.add_subplot(gs[2,2]); _ax(ax,"Final Validation Perplexity  (↓ better)")
    names=["MicroGPT","Classical","Hybrid-Q"]
    pv=[ppls["microgpt"],ppls["classical"],ppls["hybrid"]]
    bars=ax.bar(names,pv,color=[C_MG,C_CL,C_HQ],alpha=.9,width=.5)
    for bar,ppl in zip(bars,pv):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.3,f"{ppl:.2f}",
                ha="center",va="bottom",fontsize=9,color=TEXT,fontweight="bold")
    ref=ppls["classical"]
    ax.axhline(ref*1.02,ls=":",color=ORNG,lw=1.2); ax.axhline(ref*0.98,ls=":",color=ORNG,lw=1.2)
    ax.text(2.55,ref*1.02+.2,"±2%",fontsize=7,color=ORNG)
    ax.set_ylabel("Perplexity",color=TEXT)

    fig.suptitle(
        f"MicroGPT  vs  Classical MHA  vs  Hybrid Quantum Attention  ·  names.txt  ·  "
        f"d={N_EMBD}  heads={N_HEAD}  n_qubits={N_QUBITS}  steps={TRAIN_STEPS}",
        fontsize=12,fontweight="bold",color=TEXT,y=0.97)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.savefig(path,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(); print(f"\n  Figure → {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n"+"█"*68)
    print("  Three-Way Comparison: MicroGPT · Classical · Hybrid-Q")
    print("  Dataset: names.txt  Task: character-level LM")
    print("█"*68)

    # 1. MicroGPT
    print("\n"+"─"*68)
    print("  [1/3] MicroGPT  (pure-Python autograd, verbatim microgpt.py logic)")
    print("─"*68)
    mg=MicroGPT(); mg_l=[]; mg_vp=[]; t0=time.perf_counter()
    for step in range(TRAIN_STEPS):
        doc=TR_DOCS[step%len(TR_DOCS)]
        loss=mg.train_step(doc,step)
        mg_l.append(loss)
        if (step+1)%25==0:
            vp=mg.val_ppl(VAL_DOCS)
            mg_vp.append(vp)
            print(f"    [microgpt ] step {step+1:3d}/{TRAIN_STEPS}  loss={loss:.4f}  val_ppl={vp:.2f}")
    t_mg=time.perf_counter()-t0
    print(f"  MicroGPT: {t_mg:.1f}s total")

    # 2. Classical
    print("\n"+"─"*68)
    print("  [2/3] Classical MultiHeadAttention  (PyTorch, attention.py style)")
    print("─"*68)
    cl=ClassicalLM(); t0=time.perf_counter()
    cl_l,cl_vp=train_torch(cl,"classical",TRAIN_STEPS)
    t_cl=time.perf_counter()-t0
    print(f"  Classical: {t_cl:.1f}s total")

    # 3. Hybrid-Q
    print("\n"+"─"*68)
    print("  [3/3] Hybrid Quantum Attention  (Qiskit + PyTorch)")
    print("─"*68)
    hq=HybridLM(); t0=time.perf_counter()
    hq_l,hq_vp=train_hybrid(hq,TRAIN_STEPS)
    t_hq=time.perf_counter()-t0
    print(f"  Hybrid-Q: {t_hq:.1f}s total")

    # 4. Benchmark
    bench=benchmark_qkt()

    # 5. Crossover
    print("\n  Running crossover analysis …")
    cross=crossover_analysis()

    # 6. Final PPL
    print("\n  Computing final validation perplexities …")
    ppls={"microgpt":mg.val_ppl(VAL_DOCS),
          "classical":cl.val_ppl(VAL_DOCS),
          "hybrid":hq.val_ppl(VAL_DOCS)}

    # 7. Samples
    mg_s=mg.generate(8); cl_s=cl.generate(8); hq_s=hq.generate(8)

    # 8. Heatmaps on "olivia"
    probe="olivia"
    mg_h,mg_toks=mg.heat(probe)
    cl_h,cl_toks=cl.heat(probe)
    hq_h,hq_toks=hq.heat(probe)
    L=min(mg_h.shape[0],cl_h.shape[0],hq_h.shape[0])
    mg_h=mg_h[:L,:L]; cl_h=cl_h[:L,:L]; hq_h=hq_h[:L,:L]
    heat_toks=list(mg_toks[:L])

    # 9. Figure
    out="/mnt/user-data/outputs/threeway_comparison.png"
    make_figure(mg_l,cl_l,hq_l, mg_vp,cl_vp,hq_vp,
                bench,cross,
                mg_h,cl_h,hq_h,heat_toks,
                mg_s,cl_s,hq_s,ppls,out)

    # 10. Summary
    ref=ppls["classical"]
    d_arr,tc_arr,tp_arr=cross
    co_d=next((d for d,tc,tp in zip(d_arr,tc_arr,tp_arr) if tp<tc),None)
    co_red=0
    if co_d:
        idx=d_arr.index(co_d)
        co_red=(tc_arr[idx]-tp_arr[idx])/tc_arr[idx]*100

    print("\n"+"═"*68)
    print("  RESULTS SUMMARY")
    print("═"*68)
    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │  Model       │  Val PPL  │  vs Classical   │  Wall-clock (s)    │
  ├──────────────────────────────────────────────────────────────────┤
  │  MicroGPT    │  {ppls['microgpt']:7.2f}  │  {abs(ppls['microgpt']-ref)/ref*100:+7.2f}%        │  {t_mg:6.1f}               │
  │  Classical   │  {ppls['classical']:7.2f}  │  (baseline)     │  {t_cl:6.1f}               │
  │  Hybrid-Q    │  {ppls['hybrid']:7.2f}  │  {abs(ppls['hybrid']-ref)/ref*100:+7.2f}%        │  {t_hq:6.1f}               │
  └──────────────────────────────────────────────────────────────────┘

  Q·Kᵀ compute (L=16):
    MicroGPT  : {bench['microgpt_ms'][-1]:.3f} ms   (pure-Python Value scalars)
    Classical : {bench['classical_ms'][-1]:.4f} ms  (BLAS matmul)
    QHW proj  : {bench['proj_hw_ms'][-1]:.4f} ms  (projected IBM Eagle r3)

  Quantum circuit: depth={1}  gates=RY×{N_QUBITS}  n_qubits={N_QUBITS} = log₂({N_EMBD})
  Crossover at d_model = {co_d:,}  → {co_red:.1f}% faster than classical

  Generated names:
    MicroGPT  → {mg_s[:4]}
    Classical → {cl_s[:4]}
    Hybrid-Q  → {hq_s[:4]}
""")

if __name__=="__main__":
    main()