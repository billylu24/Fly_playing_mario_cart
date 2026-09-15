"""Isolated COO/CSR/edge-only backward comparisons; synthetic data only."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).with_name("sparse_operator_benchmark.json")


def case(kind,n,degree,batch,steps):
    import torch
    from sparse_edge_ops import Graph, multiply
    torch.manual_seed(42);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.55)
    row=torch.arange(n,device="cuda").repeat_interleave(degree)
    col=(row+torch.arange(1,degree+1,device="cuda").repeat(n))%n
    sparse=torch.sparse_coo_tensor(torch.stack([row,col]),torch.randn(n*degree,device="cuda")*0.2/degree**0.5,(n,n)).coalesce()
    graph=Graph(sparse.indices(),n)
    values=torch.nn.Parameter(sparse.values().clone())
    del sparse,row,col
    encoder=torch.nn.Linear(64,128,device="cuda")
    decoder=torch.nn.Linear(128,8,device="cuda")
    params=[values,*encoder.parameters(),*decoder.parameters()]
    opt=torch.optim.Adam(params,lr=1e-4,foreach=False)
    inputs=torch.randn(steps,batch,64,device="cuda")
    input_ids=torch.arange(128,device="cuda")
    output_ids=torch.arange(n-128,n,device="cuda")
    records=[]
    for iteration in range(4):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        # Native sparse tensor constructed once per unroll, matching prior benchmark.
        w=(graph.csr(values) if kind=="csr" else
           torch.sparse_coo_tensor(graph.indices,values,(n,n),is_coalesced=True) if kind=="coo" else None)
        h=torch.zeros(n,batch,device="cuda");loss=h.sum()
        for t in range(steps):
            z=encoder(inputs[t]).T
            recurrent=multiply(kind,values,h,graph) if kind in ["edge","edge_triton"] else torch.sparse.mm(w,h)
            h=0.5*h+0.5*torch.tanh(recurrent.index_add(0,input_ids,z))
            logits=decoder(h.index_select(0,output_ids).T)
            loss=loss+logits.square().mean()/steps
        torch.cuda.synchronize();fwd=time.perf_counter()
        loss.backward()
        torch.cuda.synchronize();bwd=time.perf_counter()
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in params)
        norms={"edges":values.grad.norm().item(),"encoder":encoder.weight.grad.norm().item()}
        assert all(v>0 for v in norms.values())
        opt.step();torch.cuda.synchronize();end=time.perf_counter()
        records.append({"forward_ms":(fwd-start)*1000,"backward_ms":(bwd-fwd)*1000,
                        "update_checks_ms":(end-bwd)*1000,"total_ms":(end-start)*1000,
                        "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                        "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20})
        del w,h,loss,logits,recurrent,z
    return {"status":"ok","gradient_norms":norms,"runs":records,
            "median":{k:statistics.median(r[k] for r in records[1:]) for k in records[0]}}


def suite(extra=False):
    import torch
    configs=([(8192,128,4,32),(32768,128,4,32),(165122,128,1,8),(165122,128,4,32)]
             if extra else [(8192,128,4,32),(32768,128,1,8),(32768,128,4,32)])
    output=OUT.with_name("sparse_operator_extended.json") if extra else OUT
    report={"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
            "cuda":torch.version.cuda,"free_mib_start":torch.cuda.mem_get_info()[0]/2**20,
            "allocator_cap_fraction":0.55,"cases":[]}
    for n,d,b,t in configs:
        kinds=(["edge","edge_triton"] if n>32768 else ["edge_triton"]) if extra else ["coo","csr","edge"]
        for kind in kinds:
            print(f"Running {kind} N={n} E={n*d} B={b} T={t}",flush=True)
            try:
                cp=subprocess.run([sys.executable,__file__,"--case",kind,str(n),str(d),str(b),str(t)],capture_output=True,text=True,timeout=180)
                result=json.loads(cp.stdout.strip().splitlines()[-1]) if cp.returncode==0 else {"status":"error","stderr":cp.stderr[-3000:]}
            except subprocess.TimeoutExpired:
                result={"status":"timeout","seconds":180}
            result.update(kind=kind,n=n,edges=n*d,degree=d,batch=b,steps=t)
            report["cases"].append(result);output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps({k:v for k,v in result.items() if k!="runs"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--case",nargs=5);p.add_argument("--extra",action="store_true");a=p.parse_args()
    if a.case:
        try: print(json.dumps(case(a.case[0],*map(int,a.case[1:]))))
        except Exception as e: print(json.dumps({"status":"error","error_type":type(e).__name__,"error":str(e)}))
    else:suite(a.extra)
