"""Read-only architecture transport and on-policy PPO gradient audit; no optimizer steps."""
import json,sys,math,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.model import GainPolicy
from training.task import ROOT
from training.train_gain import Environments,gae

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def transport(model):
    with torch.no_grad():
        reach=model.visual[:,None].clone();dist=torch.full((model.n,),-1,device='cuda',dtype=torch.long);dist[reach[:,0]>0]=0
        for step in range(1,33):
            nxt=(torch.sparse.mm(model.w,reach)>0).float();fresh=(nxt[:,0]>0)&(dist<0);dist[fresh]=step
            reach=torch.maximum(reach,nxt)
            if not fresh.any():break
        d=dist[model.readout];counts={str(int(k)):int((d==k).sum()) for k in d.unique()}
        rowsum=torch.sparse.mm(model.w,torch.ones(model.n,1,device='cuda'))[:,0]
        # One-edge response from visual latent channels to motor readout nodes.
        inj=torch.zeros(model.n,64,device='cuda');inj[torch.arange(model.n,device='cuda'),model.channels]=model.visual
        one=torch.sparse.mm(model.w,inj)[model.readout]
        s=torch.linalg.svdvals(one);energy=s.square();energy/=energy.sum()
        mass=one.sum(1);valid=mass>0;probs=one[valid]/mass[valid,None]
        return dict(nodes=model.n,edges=model.e,visual_nodes=int(model.visual.sum()),readout_nodes=len(model.readout),
                    visual_readout_overlap=int(model.visual[model.readout].sum()),minimum_hops_histogram=counts,
                    maximum_row_sum=float(rowsum.max()),one_hop_nonzero_readouts=int(valid.sum()),
                    one_hop_channel_entropy_mean=float(-(probs*probs.clamp_min(1e-30).log()).sum(1).mean()),
                    maximum_channel_entropy=math.log(64),one_hop_singular_energy_top1=float(energy[0]),
                    one_hop_singular_energy_top4=float(energy[:4].sum()),
                    one_hop_singular_energy_top16=float(energy[:16].sum()))

def gradient_audit(model,cfg,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    task_cfg=dict(cfg['task_config']);task_cfg['seed']=seed
    envs=Environments(4,task_cfg);frames=[];starts_list=[];acts=[];rews=[];dones=[];values=[];old=[];episodes=[]
    try:
        obs=torch.as_tensor(np.stack(envs.initial),device='cuda');h=torch.zeros(model.n,4,device='cuda');starts=torch.ones(4,device='cuda')
        with torch.no_grad():
            for t in range(32):
                frames.append(obs);starts_list.append(starts)
                logits,v,h=model(obs,h,starts);dist=Categorical(logits=logits);action=dist.sample()
                acts.append(action);values.append(v);old.append(dist.log_prob(action));msgs=envs.step(action.cpu().tolist())
                rews.append(torch.tensor([m['reward'] for m in msgs],device='cuda'));starts=torch.tensor([float(m['done']) for m in msgs],device='cuda');dones.append(starts)
                obs=torch.as_tensor(np.stack([m['frame'] for m in msgs]),device='cuda');episodes.extend(m['episode'] for m in msgs if m['episode'])
            _,bootstrap,_=model(obs,h,starts)
            rew=torch.stack(rews);oldval=torch.stack(values);adv,returns=gae(rew,oldval,torch.stack(dones),bootstrap,cfg['gamma'],cfg['gae_lambda'])
            raw_std=float(adv.std(unbiased=False));adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        h=torch.zeros(model.n,4,device='cuda');logs=[];preds=[];entropy=[]
        for o,s,a in zip(frames,starts_list,acts):
            logits,v,h=model(o,h,s);dist=Categorical(logits=logits);logs.append(dist.log_prob(a));preds.append(v);entropy.append(dist.entropy())
        lp=torch.stack(logs);ol=torch.stack(old);error=float((lp.detach()-ol).abs().max());assert error<2e-5
        ratio=(lp-ol).exp();policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        value=.25*(torch.stack(preds)-returns).square().mean();ent=-cfg['entropy_coef']*torch.stack(entropy).mean()
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for n,p in named];parts={}
        for label,loss in [('actor',policy),('critic',value),('entropy',ent)]:
            gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            parts[label]={n:(g.detach() if g is not None else torch.zeros_like(p)) for (n,p),g in zip(named,gs)}
        result=dict(seed=seed,decisions=128,completed_episodes=episodes,positive_reward_fraction=float((rew>0).float().mean()),
                    replay_error=error,raw_advantage_std=raw_std,losses=dict(actor=float(policy.detach()),critic=float(value.detach()),entropy=float(ent.detach())),modules={})
        for module in ['encoder','a','actor','critic']:
            vec={label:torch.cat([g.flatten() for n,g in gs.items() if n==module or n.startswith(module+'.')]) for label,gs in parts.items()}
            norms={label:float(g.norm()) for label,g in vec.items()};combined=sum(vec.values())
            result['modules'][module]=dict(gradient_norms=norms,actor_critic_cosine=float(torch.nn.functional.cosine_similarity(vec['actor'][None],vec['critic'][None])),combined_norm=float(combined.norm()))
        total_norm=math.sqrt(sum(x['combined_norm']**2 for x in result['modules'].values()))
        result['global_gradient_norm']=total_norm;result['global_clip_scale']=min(1,.5/(total_norm+1e-6))
        result['action_histogram']=torch.bincount(torch.stack(acts).flatten(),minlength=3).tolist()
        return result
    finally:envs.close()

def main():
    torch.set_num_threads(4);torch.manual_seed(0);out=ROOT/'tests/architecture_rl_audit';out.mkdir(exist_ok=True)
    run=ROOT/'runs/bend_gain_32k_v1';cfg=json.loads((run/'config.json').read_text());model=GainPolicy(actions=3)
    report=dict(transport=transport(model),checkpoints={},gradients=[],config=cfg)
    print('TRANSPORT',json.dumps(report['transport']),flush=True)
    for file in ['best_validation.pt','last.pt']:
        path=run/file;report['checkpoints'][file]=digest(path);model.load_state_dict(torch.load(path,weights_only=False)['model'])
        gain=.5+model.a.detach().sigmoid();bound=.5+.5*float(gain.max())*report['transport']['maximum_row_sum']
        report[file]=dict(max_gain=float(gain.max()),hidden_infinity_norm_lipschitz_bound=bound,perturbation_half_life_decisions=math.log(.5)/math.log(bound),bound_after_32_decisions=bound**32,bound_after_64_decisions=bound**64)
        for seed in [7200,7201,7202]:
            r=gradient_audit(model,cfg,seed);report['gradients'].append(dict(checkpoint=file,**r));print('GRADIENT',file,seed,json.dumps(r['modules']['encoder']),flush=True)
            (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    report['checkpoint_hashes_unchanged']=all(digest(run/f)==h for f,h in report['checkpoints'].items());assert report['checkpoint_hashes_unchanged']
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
