"""Fixed connectome with bounded per-neuron gain and cached transpose."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class FixedMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,w,wt):
        ctx.wt=wt
        return torch.sparse.mm(w,h)
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,g):return torch.sparse.mm(ctx.wt,g.contiguous()),None,None


class GainPolicy(nn.Module):
    def __init__(self,actions=7,train_gain=True):
        super().__init__()
        root=Path(__file__).resolve().parents[1]
        with np.load(root/'data/malecns/traced_smoke_graph.npz') as d:
            self.n=len(d['body_ids']);self.e=len(d['values'])
            rows=torch.tensor(d['rows'],device='cuda',dtype=torch.int64)
            cols=torch.tensor(d['cols'],device='cuda',dtype=torch.int64)
            values=torch.tensor(d['values'],device='cuda')
            crow=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int64),torch.bincount(rows,minlength=self.n).cumsum(0)])
            w=torch.sparse_csr_tensor(crow,cols,values,(self.n,self.n),check_invariants=False)
            perm=torch.argsort(cols*self.n+rows)
            tcrow=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int64),torch.bincount(cols,minlength=self.n).cumsum(0)])
            wt=torch.sparse_csr_tensor(tcrow,rows[perm],values[perm],(self.n,self.n),check_invariants=False)
            self.register_buffer('w',w,persistent=False);self.register_buffer('wt',wt,persistent=False)
            self.register_buffer('visual',torch.tensor(d['visual_mask'],device='cuda',dtype=torch.float32),persistent=False)
            self.register_buffer('readout',torch.tensor(d['readout_ids'],device='cuda'),persistent=False)
        self.register_buffer('channels',torch.arange(self.n,device='cuda')%64,persistent=False)
        self.a=nn.Parameter(torch.zeros(self.n,device='cuda'),requires_grad=train_gain)
        self.encoder=nn.Sequential(nn.Conv2d(1,16,8,4),nn.ReLU(),nn.Conv2d(16,32,4,2),nn.ReLU(),
                                   nn.Flatten(),nn.Linear(2592,64),nn.Tanh()).cuda()
        self.actor=nn.Linear(len(self.readout),actions,device='cuda');self.critic=nn.Linear(len(self.readout),1,device='cuda')
    def forward(self,obs,h,starts):
        h=h*(1-starts[None,:])
        z=self.encoder(obs[:,None].float()/255)
        signal=z.T.index_select(0,self.channels)*self.visual[:,None]
        h=0.5*h+0.5*torch.tanh((0.5+self.a.sigmoid())[:,None]*FixedMultiply.apply(h,self.w,self.wt)+signal)
        out=h.index_select(0,self.readout).T
        return self.actor(out),self.critic(out).squeeze(-1),h
