"""Separate neural internal updates from the fixed eight-frame game action."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from . import supervised as base
from .model import GainPolicy,FixedMultiply
from .task import ROOT
ORIGINAL=ROOT/'runs/supervised_architecture_v1'
OUTPUT=ROOT/'runs/neural_microsteps_v1'
STEPS=1

class MicroPolicy(GainPolicy):
    def __init__(self,steps):
        if steps not in (1,2,4):raise ValueError('steps must be 1, 2 or 4')
        super().__init__(actions=3);self.steps=steps;self.critic.requires_grad_(False)
    def forward(self,obs,h,starts):
        h=h*(1-starts[None,:]);z=self.encoder(obs[:,None].float()/255)
        signal=z.T.index_select(0,self.channels)*self.visual[:,None]
        gain=(.5+self.a.sigmoid())[:,None]
        for _ in range(self.steps):h=.5*h+.5*torch.tanh(gain*FixedMultiply.apply(h,self.w,self.wt)+signal)
        out=h.index_select(0,self.readout).T
        return self.actor(out),self.critic(out).squeeze(-1),h

def factory(arch,seed):
    assert arch=='graph';torch.manual_seed(seed);np.random.seed(seed);m=MicroPolicy(STEPS)
    return m,base.tensor_hash(m.encoder.state_dict())

def configure(steps):
    global STEPS
    STEPS=steps;base.EXPERIMENT=OUTPUT/f'k{steps}';base.make_model=factory

def prepare():
    OUTPUT.mkdir(exist_ok=False)
    protocol=dict(candidates=[2,4],baseline=1,pilot_seed=0,updates=256,
        replication_seeds=[1,2],selection='Best selected-checkpoint validation normal balanced accuracy; tie fewer internal steps.',
        gate='Replicate only if pilot validation normal BA >= archived baseline seed0 best BA + .05 AND normal BA >= frozen BA + .05.',
        rule='Gate and candidate selection before any test evaluation; no budget extensions if gate fails.',
        control='same source data, old input mapping, original initial model tensors, sample order, optimizer and game repeat8',
        warning='Internal updates also change contraction and gradient depth, not pure latency. Diagnostic test set reused.',
        source_sha256=base.digest(Path(__file__)),training_source_sha256=base.digest(Path(base.__file__)))
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for k in (2,4):
        p=OUTPUT/f'k{k}';p.mkdir();(p/'data').symlink_to(ORIGINAL/'data',target_is_directory=True)
        for file in ['manifest.json','data_summary.json']:shutil.copyfile(ORIGINAL/file,p/file)
        plan=json.loads((ORIGINAL/'plan.json').read_text());plan.update(internal_steps=k,protocol_sha256=base.digest(OUTPUT/'protocol.json'),
            sampling_unit='game decision; BPTT32 decisions implies 32*k neural updates; burn-in64 decisions',
            source_sha256=protocol['source_sha256'],test_warning=protocol['warning'])
        (p/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('PREPARED',json.dumps(protocol),flush=True)

def checks():
    torch.manual_seed(41);old=GainPolicy(actions=3);torch.manual_seed(42)
    obs=torch.randint(0,256,(2,84,84),device='cuda',dtype=torch.uint8);h=torch.randn(old.n,2,device='cuda')*.05;starts=torch.tensor([1.,0.],device='cuda')
    results={}
    for k in (1,2,4):
        model=MicroPolicy(k);model.load_state_dict(old.state_dict())
        with torch.no_grad():
            l,_,actual=model(obs,h,starts);expected=h
            for j in range(k):ref,_,expected=old(obs,expected,starts if j==0 else torch.zeros_like(starts))
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6);torch.testing.assert_close(l,ref,rtol=1e-5,atol=2e-6)
        x=obs.float().requires_grad_();logits,_,_=model(x,torch.zeros_like(h),torch.ones_like(starts))
        grad=torch.autograd.grad(logits[:,0].sum(),x)[0];norm=float(grad.norm())
        if k==1:assert norm==0
        else:assert np.isfinite(norm) and norm>0
        # Verify decision-level burn-in/refresh for this internal step count.
        rows=[dict(obs=obs[0].expand(68,-1,-1),labels=torch.arange(68,device='cuda')%3)]
        with torch.no_grad():
            batch,labels=base.batch_sequence(rows,[(0,64)],model);context=base.Context(model)
            expected=torch.stack([context.step(o) for o in rows[0]['obs']])
            torch.testing.assert_close(batch,expected[64:],rtol=1e-4,atol=2e-6)
        results[str(k)]=dict(forward_matches_repeated_reference=True,context_matches=True,current_frame_logit_gradient_norm=norm)
        del model
    (OUTPUT/'checks.json').write_text(json.dumps(results,indent=2)+'\n');print('CHECKS',json.dumps(results),flush=True)

def matched(steps,seed):
    path=OUTPUT/f'k{steps}'/f'graph_seed{seed}';ref=ORIGINAL/f'graph_seed{seed}'
    r=json.loads((path/'status.json').read_text());old=json.loads((ref/'status.json').read_text())
    assert r['valid_presentations']==old['valid_presentations'] and r['fixed_w_unchanged']
    init=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)['model'];prior=torch.load(ref/'initial.pt',map_location='cpu',weights_only=False)['model']
    assert init.keys()==prior.keys()
    for key in init:torch.testing.assert_close(init[key],prior[key],rtol=0,atol=0)
    (path/'matched_checks.json').write_text(json.dumps(dict(initial_tensors_identical=True,data_presentations_matched=True,internal_steps=steps),indent=2)+'\n')

def train(steps,seed):configure(steps);base.train('graph',seed);matched(steps,seed)

def run():
    checks()
    pilots={}
    baseline=max(r['validation']['balanced_accuracy'] for r in json.loads((ORIGINAL/'graph_seed0/metrics.json').read_text())[1:])
    for k in (2,4):
        train(k,0);model,_=factory('graph',0);ck=torch.load(base.EXPERIMENT/'graph_seed0/best.pt',weights_only=False);model.load_state_dict(ck['model'])
        rows=base.dataset('validation');pilots[str(k)]={mode:base.offline(model,rows,mode) for mode in ['normal','frozen']};del model
    chosen=max((2,4),key=lambda k:(pilots[str(k)]['normal']['balanced_accuracy'],-k))
    normal=pilots[str(chosen)]['normal']['balanced_accuracy'];frozen=pilots[str(chosen)]['frozen']['balanced_accuracy']
    passed=normal>=baseline+.05 and normal>=frozen+.05
    selection=dict(pilots=pilots,baseline_validation_ba=baseline,selected_steps=chosen,replication_gate_passed=passed,selected_before_test=True)
    (OUTPUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('SELECTION',json.dumps(selection),flush=True)
    if passed:
        for seed in [1,2]:train(chosen,seed)
    for k,seed in [(2,0),(4,0)]+([(chosen,1),(chosen,2)] if passed else []):
        configure(k);base.evaluate('graph',seed)
    (OUTPUT/'status.json').write_text(json.dumps(dict(state='completed',selection=selection),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','check','run']);a=p.parse_args();torch.set_num_threads(4)
    if a.phase=='prepare':prepare()
    elif a.phase=='check':checks()
    else:run()
if __name__=='__main__':main()
