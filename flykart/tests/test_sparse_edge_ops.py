"""Compare first-order sparse gradients to dense math and finite differences."""
import json
from pathlib import Path
import torch
from sparse_edge_ops import Graph, multiply

torch.manual_seed(123)
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.55)
results = []
for dtype in [torch.float64, torch.float32]:
    # Empty rows, uneven degree, a self-loop, negative and zero-valued edges.
    indices = torch.tensor([[0,0,1,3,3,3,5,6], [0,3,4,0,1,6,2,1]], device="cuda")
    graph = Graph(indices, 7)
    base = torch.randn(8, device="cuda", dtype=dtype)
    base[2] = 0
    for batch in [1, 3, 4]:
        # Non-contiguous dense input should also be handled correctly.
        inputs = torch.randn(batch, 7, device="cuda", dtype=dtype).T
        upstream = torch.randn(7, batch, device="cuda", dtype=dtype)
        reference_v = base.clone().requires_grad_()
        reference_h = inputs.clone().requires_grad_()
        dense = torch.sparse_coo_tensor(indices, reference_v, (7,7)).to_dense()
        reference_y = dense @ reference_h
        reference_grads = torch.autograd.grad((reference_y * upstream).sum(), (reference_v, reference_h))
        for kind in ["coo", "csr", "edge", "edge_triton"]:
            v = base.clone().requires_grad_()
            h = inputs.clone().requires_grad_()
            y = multiply(kind, v, h, graph)
            grads = torch.autograd.grad((y * upstream).sum(), (v,h))
            tol = 1e-10 if dtype == torch.float64 else 2e-5
            for actual, expected in zip([y,*grads], [reference_y,*reference_grads]):
                torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
            results.append({"kind":kind,"dtype":str(dtype),"batch":batch,
                            "max_abs_errors":[(a-b).abs().max().item() for a,b in zip([y,*grads],[reference_y,*reference_grads])]})
    # Multi-step recurrence accumulates gradients onto shared edge parameters.
    x = torch.randn(5,7,2,device="cuda",dtype=dtype)*0.1
    ref_v=base.clone().requires_grad_(); ref_x=x.clone().requires_grad_()
    def rollout(v, inp, kind):
        h=torch.zeros_like(inp[0]); loss=h.sum()
        for t in range(5):
            recurrent=(torch.sparse_coo_tensor(indices,v,(7,7)).to_dense()@h
                       if kind=="dense" else multiply(kind,v,h,graph))
            h=0.5*h+0.5*torch.tanh(recurrent+inp[t])
            loss=loss+h.square().mean()
        return loss
    ref_loss=rollout(ref_v,ref_x,"dense")
    ref_g=torch.autograd.grad(ref_loss,(ref_v,ref_x))
    for kind in ["csr","edge","edge_triton"]:
        v=base.clone().requires_grad_(); inp=x.clone().requires_grad_()
        loss=rollout(v,inp,kind); gs=torch.autograd.grad(loss,(v,inp))
        for a,b in zip([loss,*gs],[ref_loss,*ref_g]):
            torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        results.append({"kind":kind,"dtype":str(dtype),"recurrent_steps":5,"passed":True})

v=base.double().requires_grad_(); h=torch.randn(7,2,device="cuda",dtype=torch.float64,requires_grad=True)
for kind in ["edge","edge_triton"]:
    assert torch.autograd.gradcheck(lambda a,b:multiply(kind,a,b,graph),(v,h),eps=1e-6,atol=1e-5,rtol=1e-3)
for v_grad,h_grad in [(True,False),(False,True)]:
    a=v.detach().requires_grad_(v_grad);b=h.detach().requires_grad_(h_grad)
    multiply("edge",a,b,graph).sum().backward()
    assert (a.grad is not None)==v_grad and (b.grad is not None)==h_grad
report={"status":"passed","comparisons":results,"finite_difference_gradcheck":True,
        "optional_gradient_paths":True,"torch":torch.__version__}
Path(__file__).with_name("sparse_operator_correctness.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
