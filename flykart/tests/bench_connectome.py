"""Synthetic sparse recurrent compute probe. Never reads connectome or game data.

Each isolated case performs one warmup + three synthetic optimizer steps.
This measures autograd/Adam memory, not RL learning or biological fidelity.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def case(n, degree, batch, steps, mode):
    import torch
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.55)
    device = "cuda"
    # Deterministic unique incoming edges, degree matched to a graph workload.
    # This regular synthetic graph is NOT a sampled biological circuit.
    dst = torch.arange(n, device=device).repeat_interleave(degree)
    src = (dst + torch.arange(1, degree + 1, device=device).repeat(n)) % n
    indices = torch.stack([dst, src])
    values = torch.randn(n * degree, device=device) / degree**0.5 * 0.2
    sparse = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    indices = sparse.indices()
    values = torch.nn.Parameter(sparse.values().clone(), requires_grad=mode == "full")
    del sparse, dst, src
    encoder = torch.nn.Linear(64, 128, device=device)
    decoder = torch.nn.Linear(128, 8, device=device)
    gain = torch.nn.Parameter(torch.ones(n, device=device), requires_grad=mode == "gain")
    params = list(encoder.parameters()) + list(decoder.parameters())
    if values.requires_grad:
        params.append(values)
    if gain.requires_grad:
        params.append(gain)
    optimizer = torch.optim.Adam(params, lr=1e-4, foreach=False)
    x = torch.randn(steps, batch, 64, device=device)
    input_ids = torch.arange(128, device=device)
    output_ids = torch.arange(n - 128, n, device=device)
    records = []
    for iteration in range(4):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        w = torch.sparse_coo_tensor(indices, values, (n, n), is_coalesced=True)
        h = torch.zeros(n, batch, device=device)
        loss = torch.zeros((), device=device)
        for t in range(steps):
            with torch.set_grad_enabled(mode != "detach"):
                z = encoder(x[t]).T
                recurrent = torch.sparse.mm(w, h)
                h = 0.5 * h + 0.5 * torch.tanh(
                    (recurrent * gain[:, None]).index_add(0, input_ids, z))
            # Detach is a comparison: encoder receives no policy-loss gradient.
            readout = h.detach() if mode == "detach" else h
            logits = decoder(readout.index_select(0, output_ids).T)
            loss = loss + logits.square().mean() / steps
        torch.cuda.synchronize()
        forward_end = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_end = time.perf_counter()
        grads = {"encoder": encoder.weight.grad is not None,
                 "decoder": decoder.weight.grad is not None,
                 "edges": values.grad is not None, "gain": gain.grad is not None}
        assert grads["encoder"] == (mode != "detach")
        assert grads["edges"] == (mode == "full")
        assert grads["gain"] == (mode == "gain")
        assert torch.isfinite(loss).item()
        assert all(p.grad is None or torch.isfinite(p.grad).all().item() for p in params)
        gradient_norms = {"encoder": None if encoder.weight.grad is None else encoder.weight.grad.norm().item(),
                          "decoder": decoder.weight.grad.norm().item(),
                          "edges": None if values.grad is None else values.grad.norm().item(),
                          "gain": None if gain.grad is None else gain.grad.norm().item()}
        optimizer.step()
        torch.cuda.synchronize()
        end = time.perf_counter()
        records.append({"forward_ms": (forward_end-start)*1000,
                        "backward_ms": (backward_end-forward_end)*1000,
                        "update_and_checks_ms": (end-backward_end)*1000,
                        "total_ms": (end-start)*1000,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20})
        del w, h, loss, logits, recurrent, z, readout
    return {"status": "ok", "n": n, "edges": n*degree, "degree": degree,
            "batch": batch, "steps": steps, "mode": mode,
            "trainable_parameters": sum(p.numel() for p in params),
            "gradient_present": grads,
            "last_gradient_norms": gradient_norms,
            "median": {k: statistics.median(r[k] for r in records[1:]) for k in records[0]},
            "max_peak_allocated_mib_including_warmup": max(r["peak_allocated_mib"] for r in records),
            "runs": records}


def run_suite():
    import torch
    report = {"torch": torch.__version__, "cuda_build": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(),
              "capability": torch.cuda.get_device_capability(),
              "device_total_mib": torch.cuda.get_device_properties(0).total_memory/2**20,
              "free_mib_at_start": torch.cuda.mem_get_info()[0]/2**20,
              "dtype": "float32", "memory_fraction_cap": 0.55,
              "cases": []}
    configs = []
    for n, batch, steps in [(2048,1,8), (8192,1,8), (8192,1,32),
                            (8192,4,32), (32768,1,8), (32768,1,32)]:
        for mode in ["frozen", "gain", "full"]:
            configs.append((n,128,batch,steps,mode))
    configs.append((8192,128,4,32,"detach"))
    output = ROOT / "tests" / "connectome_benchmark.json"
    for n, degree, batch, steps, mode in configs:
        args = [sys.executable, __file__, "--case", str(n), str(degree), str(batch), str(steps), mode]
        print(f"Running N={n} E={n*degree} B={batch} T={steps} {mode}", flush=True)
        try:
            completed = subprocess.run(args, capture_output=True, text=True, timeout=120)
            if completed.returncode == 0:
                result = json.loads(completed.stdout.strip().splitlines()[-1])
            else:
                result = {"status": "error", "stderr": completed.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            result = {"status": "timeout", "timeout_seconds": 120}
        result.update(n=n, edges=n*degree, degree=degree, batch=batch, steps=steps, mode=mode)
        report["cases"].append(result)
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ["runs", "stderr"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=5)
    args = parser.parse_args()
    if args.case:
        try:
            print(json.dumps(case(*map(int,args.case[:4]),args.case[4])))
        except Exception as exc:
            print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}))
    else:
        run_suite()
