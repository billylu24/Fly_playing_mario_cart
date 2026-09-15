"""Task, episode boundaries, finite-horizon GAE, gain and checkpoint gates."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from training.task import Progress,Task
root=Path(__file__).resolve().parents[1]

def sample(lap,cp):return {'lap':lap+128,'current_checkpoint':cp,'lapsize':30,'getGameMode':28}
p=Progress();assert p.update(sample(-1,29))[0]<0
assert p.update(sample(0,0))[0]<0
assert p.update(sample(0,1))[0]>0
assert p.update(sample(0,0))[0]<0
assert p.update(sample(0,1))[0]<0
assert p.frontier==1
p.update(sample(0,3));assert p.frontier==1 and p.jumps==1
p.update(sample(0,2));assert p.frontier==1
p.update(sample(0,1));p.update(sample(0,2));assert p.frontier==2
for q in range(3,151):r,reason=p.update(sample(q//30,q%30))
assert reason=='success' and p.frontier==150
p=Progress(limit=1);assert p.update(sample(-1,29))[1]=='timeout'
trace=json.loads((root/'tests/track_validation/race_trace.json').read_text())
p=Progress();reward=0
for s in trace['samples']:
    r,reason=p.update(s);reward+=r
assert reason=='success' and p.frontier==150 and p.jumps==0
assert abs(reward-(10-len(trace['samples'])*0.00025))<1e-8
task=Task(limit=12)
try:
    initial=task.reset()
    for _ in range(3):m=task.step(1)
    assert m['done'] and m['episode']['reason']=='timeout'
    np.testing.assert_array_equal(m['frame'],initial)
finally:task.close()
task=Task();braking={}
try:
    for label,action in [('coast',0),('brake',4)]:
        task.reset()
        for _ in range(240):_,_,_,_,info=task.env.step(task.vectors[1])
        before_speed=info['kart1_speed']
        for _ in range(20):_,_,_,_,info=task.env.step(task.vectors[action])
        braking[label]={'before':before_speed,'after':info['kart1_speed']}
    assert braking['brake']['before']==braking['coast']['before']
    assert braking['brake']['after']<braking['coast']['after'],braking
finally:task.close()

import torch
from training.model import GainPolicy
from training.train_gain import gae
adv,ret=gae(torch.tensor([[1.],[100.]]),torch.tensor([[2.],[3.]]),torch.tensor([[1.],[0.]]),torch.tensor([4.]))
assert abs(ret[0].item()-1)<1e-6 and abs(ret[1].item()-103.96)<1e-4
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(0.55);torch.manual_seed(0)
model=GainPolicy();before=model.w.values().clone()
obs=torch.tensor(np.stack([initial,initial]),device='cuda');reset=torch.ones(2,device='cuda')
with torch.no_grad():
    a=model(obs,torch.zeros(model.n,2,device='cuda'),reset)
    b=model(obs,torch.randn(model.n,2,device='cuda'),reset)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
h=torch.zeros(model.n,2,device='cuda');loss=0
for _ in range(4):
    logits,value,h=model(obs,h,torch.zeros(2,device='cuda'));loss+=logits.square().mean()+value.square().mean()
loss.backward()
norms={name:p.grad.norm().item() for name,p in model.named_parameters()}
assert all(np.isfinite(v) and v>0 for v in norms.values())
opt=torch.optim.Adam(model.parameters(),lr=1e-4)
opt.step();assert model.a.abs().max().item()>0
assert torch.equal(model.w.values(),before) and not model.w.values().requires_grad
assert ((model.a.sigmoid()+0.5)>0.5).all() and ((model.a.sigmoid()+0.5)<1.5).all()
path=root/'tests/track_validation/gain_checkpoint_test.pt'
torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'rng':torch.get_rng_state()},path)
checkpoint=torch.load(path,weights_only=False);model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['opt'])
saved_a=model.a.detach().clone()
with torch.no_grad():model.a.add_(1)
model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['opt'])
assert torch.equal(model.a,saved_a)
torch.set_rng_state(checkpoint['rng']);expected=torch.rand(4)
torch.set_rng_state(checkpoint['rng']);assert torch.equal(torch.rand(4),expected)
report={'status':'passed','race_frames':len(trace['samples']),'race_progress':p.frontier,
        'race_reward':reward,'gain_gradient_norm':norms['a'],'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'reset_pixels':True,'hidden_reset':True,'fixed_w_unchanged':True,'checkpoint_roundtrip':True,'braking':braking}
(root/'tests/baseline_gates.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
