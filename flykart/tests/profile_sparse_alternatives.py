"""Profile the complete native-constructor and matmul gradient paths."""
import json
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity
from sparse_edge_ops import Graph, multiply

torch.manual_seed(42);torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.55)
n=4096;degree=128
r=torch.arange(n,device="cuda").repeat_interleave(degree)
c=(r+torch.arange(1,degree+1,device="cuda").repeat(n))%n
s=torch.sparse_coo_tensor(torch.stack([r,c]),torch.randn(n*degree,device="cuda"),(n,n)).coalesce()
g=Graph(s.indices(),n)
results=[]
for kind in ["csr","edge","edge_triton"]:
    v=s.values().clone().requires_grad_();h=torch.randn(n,4,device="cuda",requires_grad=True)
    y=multiply(kind,v,h,g);torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as p:
        y.square().mean().backward();torch.cuda.synchronize()
    events=[{"name":e.name,"input_shapes":e.input_shapes,"device_memory_bytes":e.device_memory_usage}
            for e in p.events() if e.name in ["aten::mm","aten::to_dense","aten::_to_dense","aten::sparse_sampled_addmm"]]
    results.append({"kind":kind,"n":n,"events":events})
    del v,h,y
Path(__file__).with_name("sparse_alternatives_profile.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results,indent=2))
