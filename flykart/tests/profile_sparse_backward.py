"""Read-only synthetic diagnosis of sparse edge-gradient intermediates."""
import json
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(0.55)
n, degree = 4096, 128
dst = torch.arange(n, device="cuda").repeat_interleave(degree)
src = (dst + torch.arange(1, degree + 1, device="cuda").repeat(n)) % n
values = torch.ones(n * degree, device="cuda", requires_grad=True)
w = torch.sparse_coo_tensor(torch.stack([dst, src]), values, (n, n),
                            check_invariants=True).coalesce()
x = torch.randn(n, 1, device="cuda", requires_grad=True)
y = torch.sparse.mm(w, x)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    y.square().mean().backward()
    torch.cuda.synchronize()
events = [{"op": e.name, "input_shapes": e.input_shapes,
           "device_memory_bytes": e.device_memory_usage}
          for e in prof.events() if e.name in ["aten::mm", "aten::to_dense", "aten::_to_dense"]]
report = {"torch": torch.__version__, "n": n, "edges": n*degree, "events": events}
text = json.dumps(report, indent=2)
print(text)
Path(__file__).with_name("sparse_backward_profile.json").write_text(text+"\n")
