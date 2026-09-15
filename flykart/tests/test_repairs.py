"""Regression tests for short-task boundaries and refreshed recurrent context."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training.task import Progress,Task
from training.model import GainPolicy

def info(cp):
    return dict(lap=128,current_checkpoint=cp,lapsize=30,getGameMode=28)
p=Progress(target=2,stall_limit=10)
p.update(info(0));p.update(info(1));r,reason=p.update(info(2))
assert reason=='target_reached' and p.frontier==2 and r>0
p=Progress(target=2,stall_limit=3)
p.update(info(0));p.update(info(0));assert p.update(info(0))[1]=='no_progress'
p=Progress(target=2,stall_limit=3,time_penalty=0)
assert p.update(info(0))[0]==0
assert p.update(info(1))[0]>0
p=Progress(target=2,stall_limit=3)
p.update(info(0));p.update(info(1));p.update(info(1));assert p.update(info(1))[1] is None
assert p.update(info(1))[1]=='no_progress'
task=Task(limit=10,repeat=8,action_set='drive')
try:
    obs=task.reset();assert len(task.vectors)==3
    assert task.step(0)['frames']==8
    m=task.step(0);assert m['done'] and m['frames']==2 and m['episode']['frames']==10
finally:task.close()
torch.set_num_threads(4);torch.manual_seed(123)
model=GainPolicy(actions=3,train_gain=False)
frames=torch.randint(0,256,(8,2,84,84),device='cuda',dtype=torch.uint8)
starts=torch.zeros(8,2,device='cuda');starts[2,0]=1
def context():
    h=torch.zeros(model.n,2,device='cuda')
    with torch.no_grad():
        for o,s in zip(frames,starts):_,_,h=model(o,h,s)
    return h
with torch.no_grad():
    h1=context();model.encoder[0].weight.add_(.0001);h2=context()
    assert not torch.equal(h1,h2)
    torch.testing.assert_close(h2,context(),rtol=1e-5,atol=1e-7)
logits,value,h=model(frames[0],h2,starts[0]);(logits.square().mean()+value.square().mean()).backward()
assert model.a.grad is None and model.actor.weight.grad is not None
assert model.encoder[0].weight.grad is not None
print('PASS: curriculum, stall, action repeat, frozen gain, fresh context replay')
