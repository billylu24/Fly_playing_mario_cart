"""Seed0 controlled pilots for spatial injection, signs and dynamics."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import supervised as base
from .model import GainPolicy,FixedMultiply
from . import layer_probes as lp
from .mechanism_audit import OUT,save
KINDS=['baseline','spatial','spatial_shuffle','gaba','gaba_shuffle','slower','gain1.2']
KIND='baseline'
SOURCE=lp.ORIGINAL.parent/'neural_microsteps_v1/k2/graph_seed0/best.pt'
class Policy(GainPolicy):
    def __init__(self,kind):
        super().__init__(actions=3);self.load_state_dict(torch.load(SOURCE,weights_only=False)['model']);self.encoder.requires_grad_(False);self.critic.requires_grad_(False)
        self.kind=kind;self.alpha=.2 if kind=='slower' else .5;self.mult=1.2 if kind=='gain1.2' else 1.
        with np.load(OUT/'metadata.npz') as d:
            signs=d['signs' if kind=='gaba' else 'shuffled_signs'] if kind in ['gaba','gaba_shuffle'] else np.ones(self.n,np.float32)
            self.register_buffer('signs',torch.tensor(signs,device='cuda'),persistent=False)
            if kind.startswith('spatial'):
                self.register_buffer('hex_ids',torch.tensor(d['hex_ids'],device='cuda'),persistent=False)
                self.register_buffer('grid',torch.tensor(d['grid' if kind=='spatial' else 'shuffled_grid'],device='cuda')[None,:,None,:],persistent=False)
                self.projection=nn.Conv2d(32,64,1,bias=False,device='cuda');nn.init.zeros_(self.projection.weight)
    def forward(self,obs,h,starts):
        h=h*(1-starts[None,:]);x=obs[:,None].float()/255
        with torch.no_grad():
            conv=self.encoder[:4](x);z=self.encoder[4:](conv)
        signal=z.T.index_select(0,self.channels)*self.visual[:,None]
        if self.kind.startswith('spatial'):
            ports=self.projection(conv)
            sampled=nn.functional.grid_sample(ports,self.grid.expand(len(obs),-1,-1,-1),align_corners=True).squeeze(-1)
            addition=sampled.gather(1,self.channels[self.hex_ids][None,None,:].expand(len(obs),1,-1)).squeeze(1).T
            signal=signal.index_add(0,self.hex_ids,addition)
        gain=(.5+self.a.sigmoid())[:,None]*self.mult
        for _ in range(2):h=(1-self.alpha)*h+self.alpha*torch.tanh(gain*FixedMultiply.apply(self.signs[:,None]*h,self.w,self.wt)+signal)
        out=h[self.readout].T;return self.actor(out),self.critic(out).squeeze(-1),h

def factory(arch,seed):
    assert arch=='graph' and seed==0;torch.manual_seed(seed);np.random.seed(seed);m=Policy(KIND);return m,base.tensor_hash(m.encoder.state_dict())
def configure(kind):
    global KIND
    KIND=kind;base.EXPERIMENT=OUT/kind;base.make_model=factory

def prepare():
    protocol=dict(candidates=KINDS,seed=0,updates=128,source_checkpoint=str(SOURCE),source_sha256=base.digest(SOURCE),frozen='CNN and critic; W absolute magnitudes/topology fixed',trainable='gain and actor in all; spatial variants additionally 2048 zero-initialized projection weights',spatial='add coordinate-sampled frozen CNN conv maps to 23720 hex nodes; paired shuffled columns within type x side; exact initial equivalence to baseline; does not isolate a pure remapping because conv representation is newly accessible',signs='conservative GABA-negative vs within-superclass/side shuffled negative neurons, fixed absolute incoming sums',dynamics='slower alpha .2 or recurrent multiplier1.2, one change each',selection='validation normal BA at update64/128, tie CE. Mechanism gate >= baseline + .05 and >= frozen + .05. Spatial/GABA additionally >= paired shuffled BA + .03. Evaluate baseline plus gate-passing candidates and their shuffled controls; no replication in this pilot.',new_test='18 predefined starts: prefix465/525/585, perturb20/40, center/left/right. Oracle feasibility retained independent of model output. New starts same track, not independent track generalization.',clock='reuse archived validation-selected time-only linear probe',source_hashes={str(p):base.digest(p) for p in [Path(__file__),Path(base.__file__),OUT/'metadata.npz']})
    save('pilot_protocol.json',protocol)
    for kind in KINDS:
        out=OUT/kind;out.mkdir(exist_ok=False);(out/'data').symlink_to(lp.ORIGINAL/'data',target_is_directory=True);shutil.copyfile(lp.ORIGINAL/'manifest.json',out/'manifest.json')
        plan=json.loads((lp.ORIGINAL/'plan.json').read_text());plan.update(updates=128,seeds=[0],experiment=kind,frozen_encoder=True);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')

def collect():
    out=OUT/'new_test';out.mkdir();manifest=[]
    for frame in [465,525,585]:
        for perturb in [20,40]:
            for turn in ['center','left','right']:
                cfg=base.make_config(frame,turn,perturb);task=base.BendTask(**cfg)
                try:
                    obs=task.reset();initial=obs.copy();observations=[];actions=[];waypoint=0
                    np.testing.assert_array_equal(initial,task.reset())
                    while True:
                        action,waypoint=base.oracle(task,waypoint);observations.append(obs);actions.append(action);msg=task.step(action);obs=msg['frame']
                        if msg['done']:break
                    accepted=bool(msg['episode']['target_reached'] and msg['episode']['jumps']==0)
                    name=cfg['start_id']+'.npz';np.savez_compressed(out/name,observations=np.stack(observations),actions=np.array(actions,np.int64));manifest.append(dict(config=cfg,accepted=accepted,filename=name,sha256=base.digest(out/name),episode=msg['episode']))
                    save('new_test_manifest.json',manifest);print('NEW_TEST',cfg['start_id'],accepted,flush=True)
                finally:task.close()

def checks():
    from .microsteps_experiment import MicroPolicy
    baseline=MicroPolicy(2);baseline.load_state_dict(torch.load(SOURCE,weights_only=False)['model']);torch.manual_seed(11);o=torch.randint(0,256,(2,84,84),device='cuda',dtype=torch.uint8);h=torch.randn(baseline.n,2,device='cuda')*.03;s=torch.tensor([1.,0.],device='cuda');checks={}
    for kind in KINDS:
        m=Policy(kind)
        with torch.no_grad():
            l,_,hh=m(o,h,s)
            if kind in ['baseline','spatial','spatial_shuffle']:
                ref,_,rh=baseline(o,h,s);torch.testing.assert_close(l,ref,rtol=1e-5,atol=2e-6);torch.testing.assert_close(hh,rh,rtol=1e-5,atol=2e-6)
        l,_,_=m(o,h,s);l.square().sum().backward();assert all(p.grad is None for p in m.encoder.parameters());assert m.actor.weight.grad.norm()>0
        if kind.startswith('spatial'):assert m.projection.weight.grad.norm()>0
        checks[kind]=dict(finite_output=bool(torch.isfinite(hh).all()),encoder_frozen=True,actor_gradient_norm=float(m.actor.weight.grad.norm()),projection_gradient_norm=float(m.projection.weight.grad.norm()) if kind.startswith('spatial') else None)
        assert checks[kind]['finite_output'];del m
    save('pilot_checks.json',checks)

def run():
    torch.set_num_threads(4);prepare();checks();collect();validation={};presentations=[]
    for kind in KINDS:
        configure(kind);base.train('graph',0);m,_=factory('graph',0);ck=torch.load(base.EXPERIMENT/'graph_seed0/best.pt',weights_only=False);m.load_state_dict(ck['model']);m.eval()
        initial=torch.load(base.EXPERIMENT/'graph_seed0/initial.pt',weights_only=False)['model']
        for key,tensor in m.encoder.state_dict().items():torch.testing.assert_close(tensor,initial['encoder.'+key],rtol=0,atol=0)
        rows=base.dataset('validation');validation[kind]={mode:base.offline(m,rows,mode) for mode in ['normal','frozen']};presentations.append(json.loads((base.EXPERIMENT/'graph_seed0/status.json').read_text())['valid_presentations']);save('pilot_validation.json',validation);print('PILOT_DONE',kind,validation[kind],flush=True);del m
    assert len(set(presentations))==1
    ba=lambda k:validation[k]['normal']['balanced_accuracy'];selected=['baseline'];gates={}
    for kind in ['spatial','gaba','slower','gain1.2']:
        passed=ba(kind)>=ba('baseline')+.05 and ba(kind)>=validation[kind]['frozen']['balanced_accuracy']+.05
        if kind in ['spatial','gaba']:passed=passed and ba(kind)>=ba(kind+'_shuffle')+.03
        gates[kind]=passed
        if passed:
            selected.append(kind)
            if kind in ['spatial','gaba']:selected.append(kind+'_shuffle')
    save('pilot_selection.json',dict(gates=gates,selected=selected,selected_before_new_test=True,matched_presentations=presentations[0]))
    # New test offline scores for all prespecified candidates after selection, no further tuning.
    manifest=json.loads((OUT/'new_test_manifest.json').read_text());test=[]
    for r in manifest:
        if r['accepted']:
            with np.load(OUT/'new_test'/r['filename']) as d:test.append(dict(meta=r,obs=torch.tensor(d['observations'],device='cuda'),labels=torch.tensor(d['actions'],device='cuda')))
    offline={}
    for kind in KINDS:
        configure(kind);m,_=factory('graph',0);m.load_state_dict(torch.load(base.EXPERIMENT/'graph_seed0/best.pt',weights_only=False)['model']);offline[kind]={mode:base.offline(m,test,mode) for mode in ['normal','frozen']};save('new_test_offline.json',offline);del m
    time=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');time.load_state_dict(torch.load(lp.OUT/'time_probe.pt',weights_only=True))
    with torch.no_grad():clock=list(time(torch.arange(225,device='cuda')[:,None]/225).softmax(-1).cpu())
    pool=base.EvaluationPool()
    try:
        for kind in selected:
            configure(kind);m,_=factory('graph',0);m.load_state_dict(torch.load(base.EXPERIMENT/'graph_seed0/best.pt',weights_only=False)['model']);report=dict(candidate=kind,episodes=[])
            for at,row in enumerate(test):
                episodes=pool.rollout(m,row['meta']['config'],clock)
                if at==0:
                    ref=base.closed_loop(m,row['meta']['config'],'normal',8100,clock);actual=next(e for e in episodes if e['mode']=='normal' and e['sample_seed']==8100)
                    for key in ['actions','frames','reason','progress','jumps']:assert ref[key]==actual[key],key
                    report['batched_sequential_match']=True
                report['episodes'].extend(episodes);save(kind+'_new_evaluation.json',report);print('EVALUATED',kind,at+1,flush=True)
            del m
    finally:pool.close()
    assert base.digest(SOURCE)==json.loads((OUT/'pilot_protocol.json').read_text())['source_sha256'];save('status.json',dict(state='completed',selected=selected,source_checkpoint_unchanged=True,matched_presentations=True,new_test_accepted=len(test)))
if __name__=='__main__':run()
