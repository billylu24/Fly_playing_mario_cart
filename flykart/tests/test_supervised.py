"""Check chunk burn-in/padding against online context, and CNN pairing."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.supervised import make_model,batch_sequence,Context,tensor_hash

def main():
    torch.set_num_threads(4);hashes=[]
    for arch in ['graph','gru']:
        model,h=make_model(arch,33);hashes.append(h)
        torch.manual_seed(987)
        rows=[dict(obs=torch.randint(0,256,(n,84,84),device='cuda',dtype=torch.uint8),labels=torch.arange(n,device='cuda')%3) for n in [75,27,95]]
        choices=[(0,32),(1,0),(2,64)]
        with torch.no_grad():
            actual,labels=batch_sequence(rows,choices,model);actual=actual.reshape(32,3,3);labels=labels.reshape(32,3)
            for b,(i,t) in enumerate(choices):
                context=Context(model);reference=torch.stack([context.step(o) for o in rows[i]['obs']])
                n=min(32,len(rows[i]['labels'])-t)
                torch.testing.assert_close(actual[:n,b],reference[t:t+n],rtol=1e-4,atol=2e-6)
                torch.testing.assert_close(labels[:n,b],rows[i]['labels'][t:t+n])
                assert (labels[n:,b]==-100).all()
        logits,labels=batch_sequence(rows,choices,model)
        torch.nn.functional.cross_entropy(logits,labels).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
        assert sum(float(p.grad.square().sum()) for p in model.encoder.parameters())>0
        print('PASS',arch,'context equivalence, padding, encoder gradient',flush=True)
        del model
    assert hashes[0]==hashes[1]
    print('PASS identical initial encoders',flush=True)
if __name__=='__main__':main()
