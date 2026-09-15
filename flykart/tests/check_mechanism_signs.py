"""Verify sender signs and gradients against an explicitly signed sparse matrix."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.model import GainPolicy,FixedMultiply
from training.mechanism_audit import OUT

def main():
    torch.set_num_threads(2);torch.manual_seed(24);m=GainPolicy(actions=3)
    with np.load(OUT/'metadata.npz') as d:sign=torch.tensor(d['signs'],device='cuda')
    signed=torch.sparse_csr_tensor(m.w.crow_indices(),m.w.col_indices(),m.w.values()*sign[m.w.col_indices()],m.w.shape,check_invariants=False)
    h=torch.randn(m.n,2,device='cuda',requires_grad=True);g=torch.randn_like(h)
    implicit=FixedMultiply.apply(sign[:,None]*h,m.w,m.wt);explicit=torch.sparse.mm(signed,h)
    torch.testing.assert_close(implicit,explicit,rtol=1e-5,atol=2e-6)
    ga=torch.autograd.grad((implicit*g).sum(),h)[0];gb=torch.autograd.grad((explicit*g).sum(),h)[0];torch.testing.assert_close(ga,gb,rtol=1e-4,atol=5e-6)
    assert torch.equal(signed.values().abs(),m.w.values())
    (OUT/'sign_operator_checks.json').write_text(json.dumps(dict(sender_sign_forward_matches=True,backward_matches_signed_sparse_autograd=True,absolute_edge_weights_unchanged=True,maximum_gradient_error=float((ga-gb).abs().max())),indent=2)+'\n')
if __name__=='__main__':main()
