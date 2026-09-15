"""Trace gradients and reconstruct frozen CNN targets from motor states."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import supervised as base
from . import layer_probes as lp
from .model import FixedMultiply
OUT=lp.OUT.parent/'transport_reconstruction_v1'
SOURCE=lp.ORIGINAL.parent/'neural_microsteps_v1/k2/graph_seed0/best.pt'
KINDS=['decoder_only','gain','interface','interface_edges']
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
class Transport(nn.Module):
    def __init__(self,backbone,kind,stats,edges):
        super().__init__();self.b=backbone;self.kind=kind;self.b.requires_grad_(False);self.b.a.requires_grad_(kind!='decoder_only');self.n=backbone.n
        for k,v in stats.items():self.register_buffer(k,v.clone())
        rng=torch.Generator(device='cuda').manual_seed(8181)
        self.register_buffer('ports',torch.randn(self.n,8,device='cuda',generator=rng)/8**.5)
        self.interface=nn.Parameter(torch.zeros(8,64,device='cuda'),requires_grad=kind in ['interface','interface_edges'])
        self.register_buffer('src',edges[0]);self.register_buffer('dst',edges[1]);self.register_buffer('edge_weight',edges[2]);self.delta=nn.Parameter(torch.zeros(len(edges[0]),device='cuda'),requires_grad=kind=='interface_edges')
        torch.manual_seed(81);self.decoder=nn.Linear(len(backbone.readout),64,device='cuda')
        nn.init.zeros_(self.decoder.weight);nn.init.zeros_(self.decoder.bias)
    def sequence(self,z,inspect=False):
        h=torch.zeros(self.n,z.shape[1],device='cuda');outputs=[];states=[];signals=[]
        for at in range(len(z)):
            signal=z[at].T.index_select(0,self.b.channels)*self.b.visual[:,None]
            if self.kind in ['interface','interface_edges']:
                signal=signal+(self.ports@(self.interface@((z[at]-self.zmean)/self.zscale).T))*self.b.visual[:,None]
            if inspect:signal.requires_grad_(True);signal.retain_grad();signals.append(signal)
            for _ in range(2):
                recurrent=FixedMultiply.apply(h,self.b.w,self.b.wt)
                if self.kind=='interface_edges':
                    addition=(.5*self.edge_weight*self.delta.tanh())[:,None]*h[self.src]
                    recurrent=recurrent.index_add(0,self.dst,addition)
                h=.5*h+.5*torch.tanh((.5+self.b.a.sigmoid())[:,None]*recurrent+signal)
                if inspect:h.retain_grad();states.append(h)
            outputs.append(self.decoder((h[self.b.readout].T-self.hmean)/self.hscale))
        return torch.stack(outputs),states,signals
@torch.no_grad()
def raw_states(b,z):
    h=torch.zeros(b.n,z.shape[1],device='cuda');xs=[]
    for zz in z:
        signal=zz.T.index_select(0,b.channels)*b.visual[:,None]
        for _ in range(2):h=.5*h+.5*torch.tanh((.5+b.a.sigmoid())[:,None]*FixedMultiply.apply(h,b.w,b.wt)+signal)
        xs.append(h[b.readout].T)
    return torch.stack(xs)
def measure(model,z,target):
    with torch.no_grad():
        pred,_,_=model.sequence(z);error=(pred-target).square();return dict(normalized_mse=float(error.mean()),r2_global=float(1-error.sum()/((target-target.mean((0,1))).square().sum().clamp_min(1e-12))))
def gradients(model,z,target):
    model.zero_grad(set_to_none=True);pred,states,signals=model.sequence(z,True);loss=(pred[-1]-target[-1]).square().mean();loss.backward()
    masks={'visual':model.b.visual.bool(),'motor':torch.zeros(model.n,device='cuda',dtype=torch.bool)};masks['motor'][model.b.readout]=True;masks['intermediate']=~(masks['visual']|masks['motor'])
    hist=[]
    for t,h in enumerate(states):
        g=h.grad
        hist.append(dict(neural_step=t+1,lag_from_final=len(states)-t-1,gradient_rms={k:float(g[mask].square().mean().sqrt()) if g is not None else 0. for k,mask in masks.items()}))
    return dict(final_decision_loss=float(loss.detach()),state_gradients=hist,injection_gradient_rms=[float(x.grad.square().mean().sqrt()) if x.grad is not None else 0. for x in signals],parameter_gradient_norm={n:float(p.grad.norm()) if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad})
def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=False)
    save('protocol.json',dict(seed=0,source=str(SOURCE),source_sha256=base.digest(SOURCE),training_ids=['f450_center_p12','f480_left_p12','f540_right_p12'],training='first24 observations of each trajectory, 72 fixed targets; full BPTT24 decisions/48 neural updates, reset at start; 400 full-batch updates Adam lr.003 eps1e-5 clip1',validation='first21 of each of six original validation trajectories, whole trajectory held out; final checkpoint only, no validation tuning',targets='frozen k2 CNN64 standardized on 72 training targets; reconstruct current frame representation',decoder='linear2129->64; zero initialization; fixed training baseline motor standardization std floor1e-5',arms=KINDS,interface='fixed seeded random N x8 basis, trainable8x64 residual using standardized target features as INPUT representation; zero initial; visual nodes only',edges='up to8 strongest existing visual->motor edges per recipient; each weight may change by +/-50% using tanh delta; no new edges',checks='all arms identical initial outputs; CNN frozen and source unchanged; nonzero gradients may not imply useful learning',reference='constant train mean and ridge linear fit to frozen motor states; ridge values chosen on training reconstruction only',fit_gate='training normalized MSE <= .05 as practical small-set fit, not driving success',source_code_sha256=base.digest(Path(__file__))))
    b,_=lp.backbone('k2');source_state=copy.deepcopy(b.state_dict());source_hash=base.tensor_hash(b.state_dict());rows=base.dataset('train');ids=json.loads((OUT/'protocol.json').read_text())['training_ids'];chosen=[next(r for r in rows if r['meta']['start_id']==i) for i in ids];validation=base.dataset('validation')
    with torch.no_grad():
        def encode(rs,length):return torch.stack([b.encoder(r['obs'][:length,None].float()/255) for r in rs],1)
        z=encode(chosen,24);vz=encode(validation,21);zm=z.mean((0,1));zs=z.std((0,1)).clamp_min(1e-5);target=(z-zm)/zs;vt=(vz-zm)/zs;hx=raw_states(b,z);hm=hx.mean((0,1));hs=hx.std((0,1)).clamp_min(1e-5);vhh=raw_states(b,vz)
    with np.load(base.ROOT/'data/malecns/traced_smoke_graph.npz') as d:
        rr=d['rows'];cc=d['cols'];ww=d['values'];visual=d['visual_mask'].astype(bool);motor=np.zeros(b.n,bool);motor[d['readout_ids']]=True;ix=np.flatnonzero(motor[rr]&visual[cc]);selected=[]
        for node in np.unique(rr[ix]):q=ix[rr[ix]==node];selected.extend(q[np.argsort(ww[q],kind='stable')[-8:]].tolist())
        ix=np.array(selected);edges=tuple(torch.tensor(a,device='cuda') for a in [cc[ix].astype(np.int64),rr[ix].astype(np.int64),ww[ix]]);np.savez(OUT/'edges.npz',src=cc[ix],dst=rr[ix],weights=ww[ix])
    stats=dict(zmean=zm,zscale=zs,hmean=hm,hscale=hs);torch.save(dict(z=z.cpu(),validation_z=vz.cpu(),stats={k:v.cpu() for k,v in stats.items()}),OUT/'features.pt')
    # Closed-form frozen readout reference, using the same features as trained decoder.
    x=((hx-hm)/hs).flatten(0,1).double();xx=torch.cat([x,torch.ones(len(x),1,device='cuda',dtype=torch.double)],1);y=target.flatten(0,1).double();vxx=torch.cat([((vhh-hm)/hs).flatten(0,1).double(),torch.ones(vt.numel()//64,1,device='cuda',dtype=torch.double)],1)
    refs={};best=None
    for ridge in [1e-8,1e-5,.001,.1,1.]:
        coef=xx.T@torch.linalg.solve(xx@xx.T+ridge*torch.eye(len(xx),device='cuda',dtype=torch.double),y);tm=float((xx@coef-y).square().mean());vm=float((vxx@coef-vt.flatten(0,1)).square().mean());refs[str(ridge)]=dict(train_mse=tm,validation_mse=vm)
        if best is None or tm<best[0]:best=(tm,ridge,coef)
    torch.save(best[2].cpu(),OUT/'ridge_decoder.pt');save('references.json',dict(ridge=refs,selected_by_training_only=best[1],constant_train_mse=float(target.square().mean()),constant_validation_mse=float(vt.square().mean()),edge_count=len(ix),training_points=target.numel()//64,validation_points=vt.numel()//64))
    results={}
    for kind in KINDS:
        b.load_state_dict(source_state);model=Transport(b,kind,stats,edges);initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.003,eps=1e-5);history=[];audits={};start=time.perf_counter()
        for step in range(401):
            if step in [0,1,25,100,200,400]:
                history.append(dict(step=step,train=measure(model,z,target),validation=measure(model,vz,vt)));audits[str(step)]=gradients(model,z,target)
                print('MEASURE',kind,step,history[-1],flush=True)
            if step==400:break
            opt.zero_grad(set_to_none=True);pred,_,_=model.sequence(z);loss=(pred-target).square().mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if (step+1)%50==0:print('TRAIN',kind,step+1,float(loss.detach()),flush=True)
        delta={n:dict(update_l2=float((p.detach()-initial[n]).norm()),initial_l2=float(initial[n].norm()),final_l2=float(p.detach().norm())) for n,p in model.named_parameters() if p.requires_grad}
        assert base.tensor_hash(b.encoder.state_dict())==base.tensor_hash({k[8:]:v for k,v in source_state.items() if k.startswith('encoder.')});assert all(p.grad is None for p in b.encoder.parameters())
        with torch.no_grad():
            frozen_z=z[:1].expand_as(z);fv=vz[:1].expand_as(vz)
            interventions=dict(train_frozen=measure(model,frozen_z,target),validation_frozen=measure(model,fv,vt))
        result=dict(history=history,gradient_audits=audits,parameter_updates=delta,interventions=interventions,wall_seconds=time.perf_counter()-start,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),encoder_unchanged=True,encoder_gradient_none=True)
        save(kind+'.json',result);torch.save(model.state_dict(),OUT/(kind+'.pt'));results[kind]=result;del model,opt
    b.load_state_dict(source_state);assert base.tensor_hash(b.state_dict())==source_hash;assert base.digest(SOURCE)==json.loads((OUT/'protocol.json').read_text())['source_sha256'];save('status.json',dict(state='completed',source_checkpoint_unchanged=True,all_encoder_checks_passed=True,training_fit_gate={k:v['history'][-1]['train']['normalized_mse']<=.05 for k,v in results.items()}))
if __name__=='__main__':run()
