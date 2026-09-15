"""Bounded GAIN pilot; recurrent PPO with burn-in and exact decision budget."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.distributions import Categorical
from .model import GainPolicy
from .task import ROOT,Task,worker,make_task


def gae(rewards,values,dones,bootstrap,gamma=.99,lam=.95):
    adv=torch.zeros_like(rewards);carry=torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        nv=bootstrap if t==len(rewards)-1 else values[t+1]
        delta=rewards[t]+gamma*nv*(1-dones[t])-values[t]
        carry=delta+gamma*lam*(1-dones[t])*carry;adv[t]=carry
    return adv,adv+values


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class Environments:
    def __init__(self,n,task_config=None):
        ctx=mp.get_context('spawn');self.pipes=[];self.processes=[]
        for i in range(n):
            cfg=dict(task_config or {})
            if cfg.get('kind')=='bend':cfg['seed']=cfg.get('seed',0)+i
            p,c=ctx.Pipe();proc=ctx.Process(target=worker,args=(c,cfg));proc.start();c.close()
            self.pipes.append(p);self.processes.append(proc)
        self.initial=[self.receive(p)['frame'] for p in self.pipes]
    @staticmethod
    def receive(p):
        if not p.poll(30):raise TimeoutError('game worker timeout')
        result=p.recv()
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def step(self,actions):
        for p,a in zip(self.pipes,actions):p.send(int(a))
        return [self.receive(p) for p in self.pipes]
    def close(self):
        for p in self.pipes:
            try:p.send('close')
            except (BrokenPipeError,EOFError):pass
        for proc in self.processes:
            proc.join(5)
            if proc.is_alive():proc.terminate();proc.join(5)
        for p in self.pipes:p.close()


def evaluate(model,seed_base=1000,episodes=10,task_config=None,memory=64,vision='normal'):
    # Separate CPU RNG avoids changing training action sampling / minibatch RNG.
    task=make_task(task_config);results=[]
    try:
        with torch.no_grad():
            for seed in range(seed_base,seed_base+episodes):
                rng=torch.Generator().manual_seed(seed);obs=task.reset()
                initial=obs.copy()
                h=torch.zeros(model.n,1,device='cuda');starts=torch.ones(1,device='cuda')
                history=[];step=0
                while True:
                    if vision=='frozen':obs=initial
                    if step%32==0:
                        h=torch.zeros_like(h)
                        for old_obs,old_start in history[-memory:]:
                            _,_,h=model(old_obs,h,old_start)
                    history.append((torch.as_tensor(obs[None],device='cuda'),starts.clone()))
                    history=history[-memory:]
                    logits,_,h=model(torch.as_tensor(obs[None],device='cuda'),h,starts)
                    action=int(torch.multinomial(logits.softmax(-1).cpu()[0],1,generator=rng))
                    m=task.step(action);obs=m['frame'];starts.fill_(0);step+=1
                    if m['done']:
                        results.append({'seed':seed,**m['episode']});break
    finally:task.close()
    return results


def train(args):
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.55)
    run=Path(args.run);run.mkdir(parents=True,exist_ok=False)
    config=vars(args).copy()
    config.update(graph_sha256=digest_file(ROOT/'data/malecns/traced_smoke_graph.npz'),
                  rom_sha256=digest_file(ROOT/'mario_kart/SuperMarioKart-Snes/rom.sfc'),
                  state_sha256=digest_file(ROOT/'mario_kart/SuperMarioKart-Snes/MarioCircuit_M.state'),
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                  source_hashes={p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')},
                  reward='frontier/30 +5*target/150 on target -time_penalty/frame',
                  learning_rate=1e-4,clip=.2,epochs=4,target_kl=.02,burn_in=args.memory,unroll=32,
                  state_policy='zero-rooted context, refreshed every 32 decisions')
    task_config=dict(target=args.target,stall_limit=args.stall_frames,repeat=args.repeat,action_set=args.action_set,time_penalty=args.time_penalty)
    if args.task=='bend':
        task_config=dict(kind='bend',start_ids=['f540_center','f540_left','f540_right','f600_center','f600_right'],
                         perturb_frames=24,seed=args.seed*100,action_set='drive')
        config['bend_actual_limits']=dict(limit=1800,target=13,stall_limit=600,repeat=8,time_penalty=0)
        config['prefix_sha256']=digest_file(ROOT/'tests/track_validation/race_trace.json')
    config['task_config']=task_config
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    model=GainPolicy(actions=7 if args.action_set=='full' else 3,train_gain=not args.freeze_gain)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,eps=1e-5)
    fixed_hash=hashlib.sha256(model.w.values().cpu().numpy().tobytes()).hexdigest()
    envs=None;decisions=0;frames=0;optimizer_steps=0;iteration=0;best_key=None;eval_history=[]
    stats=[];start_run=time.perf_counter();next_eval=args.eval_every;next_save=10000
    if args.resume:
        checkpoint=torch.load(args.resume,weights_only=False)
        for key in ['state_policy','task_config','memory','freeze_gain','gamma','gae_lambda','entropy_coef']:
            if checkpoint['config'].get(key)!=config[key]:raise ValueError(f'incompatible resume: {key}')
        assert checkpoint['config']['graph_sha256']==config['graph_sha256']
        assert checkpoint['fixed_w_sha256']==fixed_hash
        assert checkpoint['config']['envs']==args.envs and checkpoint['config']['sequences']==args.sequences
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        decisions=checkpoint['decisions'];frames=checkpoint['frames'];optimizer_steps=checkpoint['optimizer_steps'];iteration=checkpoint['iteration']
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        assert decisions<args.budget
        next_eval=((decisions//args.eval_every)+1)*args.eval_every if args.eval_every else 0
        next_save=((decisions//10000)+1)*10000
    status={'state':'initializing','decisions':0}
    def save(name):
        check=hashlib.sha256(model.w.values().cpu().numpy().tobytes()).hexdigest()
        assert check==fixed_hash,'frozen W changed'
        data={'model':model.state_dict(),'optimizer':opt.state_dict(),'decisions':decisions,
              'frames':frames,'optimizer_steps':optimizer_steps,'iteration':iteration,
              'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
              'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
              'config':config,'fixed_w_sha256':fixed_hash,
              'resume_requires_new_episode':True}
        temp=run/(name+'.tmp');torch.save(data,temp);temp.replace(run/name)
    try:
        envs=Environments(args.envs,task_config)
        obs=torch.as_tensor(np.stack(envs.initial),device='cuda')
        h=torch.zeros(model.n,args.envs,device='cuda');starts=torch.ones(args.envs,device='cuda')
        history_obs=[];history_starts=[];history_h=[]
        save('initial.pt')
        while decisions<args.budget:
            length=min(32,(args.budget-decisions)//args.envs)
            if length<=0:break
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
            # A bounded, zero-rooted context defines BOTH behavior and update states.
            # No hidden states generated by old parameters survive this refresh.
            with torch.no_grad():
                h=torch.zeros_like(h)
                for old_obs,old_start in zip(history_obs,history_starts):
                    _,_,h=model(old_obs,h,old_start)
            obs_buf=[];start_buf=[];acts=[];oldlogs=[];values=[];rewards=[];dones=[];hs=[];episodes=[]
            for t in range(length):
                hs.append(h.detach().clone());obs_buf.append(obs);start_buf.append(starts)
                with torch.no_grad():
                    logits,v,h=model(obs,h,starts);dist=Categorical(logits=logits)
                    action=dist.sample();log=dist.log_prob(action)
                msgs=envs.step(action.cpu().tolist())
                acts.append(action);oldlogs.append(log);values.append(v)
                rewards.append(torch.tensor([m['reward'] for m in msgs],device='cuda'))
                starts=torch.tensor([float(m['done']) for m in msgs],device='cuda');dones.append(starts)
                obs=torch.as_tensor(np.stack([m['frame'] for m in msgs]),device='cuda')
                decisions+=args.envs;frames+=sum(m['frames'] for m in msgs)
                episodes.extend(m['episode'] for m in msgs if m['episode'])
            with torch.no_grad():
                bootstrap_h=torch.zeros_like(h)
                for old_obs,old_start in zip((history_obs+obs_buf)[-args.memory:],(history_starts+start_buf)[-args.memory:]):
                    _,_,bootstrap_h=model(old_obs,bootstrap_h,old_start)
                _,bootstrap,_=model(obs,bootstrap_h,starts)
            ob=torch.stack(obs_buf);st=torch.stack(start_buf);actions=torch.stack(acts)
            oldlog=torch.stack(oldlogs);oldval=torch.stack(values);rew=torch.stack(rewards);done=torch.stack(dones)
            advantage,returns=gae(rew,oldval,done,bootstrap,args.gamma,args.gae_lambda)
            raw_adv_std=advantage.std(unbiased=False).item();raw_adv_mean=advantage.mean().item()
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            # Check exact behavior-state replay separately from approximate burn-in.
            with torch.no_grad():
                verify_h=hs[0];verify_logs=[]
                for t in range(length):
                    vl,_,verify_h=model(ob[t],verify_h,st[t])
                    verify_logs.append(Categorical(logits=vl).log_prob(actions[t]))
                behavior_replay_error=(torch.stack(verify_logs)-oldlog).abs().max().item()
                assert behavior_replay_error<2e-5,('behavior replay mismatch',behavior_replay_error)
                del verify_h,verify_logs,vl
            combined_obs=history_obs+obs_buf;combined_st=history_starts+start_buf;combined_h=history_h+hs
            offset=len(history_obs)
            # Short final rollouts use padding + loss mask to hit exact budget.
            sequences=[(i,t,min(32,length-t)) for i in range(args.envs) for t in range(0,length,32)]
            torch.cuda.synchronize();sample_end=time.perf_counter()
            update_logs=[];replay_error=0.;gain_grad=0.
            for epoch in range(4):
                order=np.random.permutation(len(sequences));epoch_kl_sum=0.;epoch_count=0
                for at in range(0,len(order),args.sequences):
                    seqs=[sequences[k] for k in order[at:at+args.sequences]];count=len(seqs)
                    pos=[offset+t for i,t,k in seqs];lo=[0 for p in pos]
                    states=torch.zeros(model.n,count,device='cuda')
                    # Per-sequence burn-in can have different prefix length near run start.
                    with torch.no_grad():
                        for b in range(args.memory):
                            active=torch.tensor([l+b<p for l,p in zip(lo,pos)],device='cuda')
                            if not active.any():break
                            idx=[min(l+b,p-1) if p>0 else 0 for l,p in zip(lo,pos)]
                            bo=torch.stack([combined_obs[j][i] for j,(i,t,k) in zip(idx,seqs)])
                            bs=torch.stack([combined_st[j][i] for j,(i,t,k) in zip(idx,seqs)])
                            _,_,candidate=model(bo,states,bs)
                            states=torch.where(active[None,:],candidate,states)
                    states=states.detach();newlogs=[];pred=[];ent=[];old=[];av=[];targets=[];valid=[]
                    opt.zero_grad(set_to_none=True)
                    for step in range(max(k for i,t,k in seqs)):
                        js=[min(t+step,length-1) for i,t,k in seqs]
                        batchobs=torch.stack([ob[j,i] for j,(i,t,k) in zip(js,seqs)])
                        batchstart=torch.stack([st[j,i] for j,(i,t,k) in zip(js,seqs)])
                        logits,v,states=model(batchobs,states,batchstart);dist=Categorical(logits=logits)
                        action=torch.stack([actions[j,i] for j,(i,t,k) in zip(js,seqs)])
                        newlogs.append(dist.log_prob(action));pred.append(v);ent.append(dist.entropy())
                        old.append(torch.stack([oldlog[j,i] for j,(i,t,k) in zip(js,seqs)]))
                        av.append(torch.stack([advantage[j,i] for j,(i,t,k) in zip(js,seqs)]))
                        targets.append(torch.stack([returns[j,i] for j,(i,t,k) in zip(js,seqs)]))
                        valid.append([step<k for i,t,k in seqs])
                    mask=torch.tensor(valid,device='cuda');lp=torch.stack(newlogs)[mask];ol=torch.stack(old)[mask]
                    a=torch.stack(av)[mask];target=torch.stack(targets)[mask];v=torch.stack(pred)[mask]
                    entropy=torch.stack(ent)[mask].mean();ratio=(lp-ol).exp()
                    if epoch==0 and at==0:
                        replay_error=(lp-ol).abs().max().item()
                        assert replay_error<2e-5,('context replay mismatch',replay_error)
                    policy_loss=-torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
                    value_loss=.5*(v-target).square().mean()
                    loss=policy_loss+.5*value_loss-args.entropy_coef*entropy
                    assert torch.isfinite(loss).item()
                    loss.backward()
                    gain_grad=model.a.grad.norm().item() if model.a.grad is not None else 0.
                    encoder_grad=sum(p.grad.square().sum().item() for p in model.encoder.parameters() if p.grad is not None)**.5
                    assert np.isfinite(gain_grad)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True).item()
                    opt.step();optimizer_steps+=1
                    kl=((ratio-1)-(lp-ol)).mean().item();nvalid=mask.sum().item()
                    epoch_kl_sum+=kl*nvalid;epoch_count+=nvalid
                    update_logs.append({'loss':loss.item(),'policy_loss':policy_loss.item(),'value_loss':value_loss.item(),
                                        'entropy':entropy.item(),'kl':kl,'clip_fraction':((ratio-1).abs()>.2).float().mean().item(),
                                        'grad_norm':norm})
                    del states,newlogs,pred,ent,lp,ol,a,target,v,entropy,ratio,loss,policy_loss,value_loss,dist,logits
                if epoch_kl_sum/epoch_count>.02:break
            history_obs=combined_obs[-args.memory:];history_starts=combined_st[-args.memory:];history_h=[]
            h=h.detach();torch.cuda.synchronize();end=time.perf_counter()
            gain=.5+model.a.detach().sigmoid()
            target_variance=returns.var(unbiased=False).item()
            record={'iteration':iteration,'decisions':decisions,'frames':frames,'optimizer_steps':optimizer_steps,
                    'rollout_seconds':sample_end-begin,'update_seconds':end-sample_end,'total_seconds':end-begin,
                    'decision_rate':length*args.envs/(end-begin),'episode_results':episodes,
                    'reward_sum':rew.sum().item(),'reward_max':rew.max().item(),'progress_max':max(m['progress'] for m in msgs),
                    'behavior_replay_error':behavior_replay_error,'burnin_initial_logprob_drift':replay_error,
                    'gain_grad_norm':gain_grad,'gain_min':gain.min().item(),'gain_max':gain.max().item(),
                    'encoder_grad_norm':encoder_grad,'raw_advantage_std':raw_adv_std,'raw_advantage_mean':raw_adv_mean,
                    'positive_reward_fraction':(rew>0).float().mean().item(),
                    'gain_quantiles':torch.quantile(gain,torch.tensor([.01,.5,.99],device='cuda')).tolist(),
                    'gain_near_boundary_fraction':((gain<.51)|(gain>1.49)).float().mean().item(),
                    'hidden_rms':h.square().mean().sqrt().item(),'hidden_saturated_fraction':(h.abs()>.99).float().mean().item(),
                    'explained_variance_behavior':None if target_variance<1e-10 else 1-(returns-oldval).var(unbiased=False).item()/target_variance,
                    'action_histogram':torch.bincount(actions.flatten(),minlength=7).tolist(),
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                    **{k:float(np.mean([x[k] for x in update_logs])) for k in update_logs[0]}}
            stats.append(record)
            with open(run/'metrics.jsonl','a') as f:f.write(json.dumps(record)+'\n')
            iteration+=1
            status={'state':'training','decisions':decisions,'budget':args.budget,'iteration':iteration,
                    'last_metrics':record,'wall_seconds':time.perf_counter()-start_run}
            (run/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            if iteration%10==0 or iteration==1:print(json.dumps({k:v for k,v in record.items() if k!='episode_results'}),flush=True)
            if decisions>=next_save or decisions==args.budget:
                save('last.pt');next_save+=10000
            if args.eval_every and (decisions>=next_eval or decisions==args.budget):
                save('last.pt');status['state']='evaluating';(run/'status.json').write_text(json.dumps(status,indent=2)+'\n')
                save(f'step_{decisions}.pt')
                eval_start=time.perf_counter()
                if args.task=='bend':
                    results=[];frozen=[]
                    for start_id in ['f570_center','f570_left','f570_right']:
                        ecfg=dict(kind='bend',start_id=start_id,perturb_frames=24,action_set='drive')
                        results.extend(evaluate(model,seed_base=5000,episodes=3,task_config=ecfg,memory=args.memory))
                        frozen.extend(evaluate(model,seed_base=5000,episodes=3,task_config=ecfg,memory=args.memory,vision='frozen'))
                else:results=evaluate(model,episodes=args.eval_episodes,task_config=task_config,memory=args.memory)
                evalrecord={'decisions':decisions,'wall_seconds':time.perf_counter()-eval_start,'episodes':results,
                            'success_rate':sum(x['reason']=='success' for x in results)/len(results),
                            'one_lap_rate':sum(x['one_lap'] for x in results)/len(results)}
                evalrecord['target_reached_rate']=sum(x['target_reached'] for x in results)/len(results)
                if args.task=='bend':
                    evalrecord['frozen_episodes']=frozen
                    evalrecord['frozen_target_rate']=sum(x['target_reached'] for x in frozen)/len(frozen)
                eval_history.append(evalrecord)
                with open(run/'evaluations.jsonl','a') as f:f.write(json.dumps(evalrecord)+'\n')
                successes=[x['frames'] for x in results if x['reason']=='success']
                key=(evalrecord['target_reached_rate'],-np.mean(successes) if successes else -18000,np.mean([x['progress'] for x in results]))
                if best_key is None or key>best_key:best_key=key;save('best_validation.pt')
                print('EVALUATION',json.dumps(evalrecord),flush=True);next_eval+=args.eval_every
            del obs_buf,start_buf,acts,oldlogs,values,rewards,dones,hs,ob,st,actions,oldlog,oldval,rew,done
            del combined_obs,combined_st,combined_h,advantage,returns
        assert decisions==args.budget
        save('last.pt')
        status={'state':'completed','decisions':decisions,'frames':frames,'optimizer_steps':optimizer_steps,
                'wall_seconds':time.perf_counter()-start_run,'evaluations':eval_history,
                'median_decisions_per_second':float(np.median([r['decision_rate'] for r in stats[1:] or stats])),
                'max_allocated_mib':max(r['peak_allocated_mib'] for r in stats),'fixed_w_unchanged':True}
        (run/'status.json').write_text(json.dumps(status,indent=2)+'\n');print('COMPLETE',json.dumps(status),flush=True)
    except BaseException as exc:
        save('interrupted.pt')
        status.update(state='failed',decisions=decisions,error=repr(exc))
        (run/'status.json').write_text(json.dumps(status,indent=2)+'\n');raise
    finally:
        if envs:envs.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--budget',type=int,default=200000)
    p.add_argument('--envs',type=int,default=4);p.add_argument('--sequences',type=int,default=4)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--eval-every',type=int,default=25000)
    p.add_argument('--resume',default=None)
    p.add_argument('--task',choices=['race','bend'],default='race')
    p.add_argument('--memory',type=int,default=64)
    p.add_argument('--target',type=int,default=150)
    p.add_argument('--stall-frames',type=int,default=0)
    p.add_argument('--time-penalty',type=float,default=.00025)
    p.add_argument('--repeat',type=int,default=4)
    p.add_argument('--action-set',choices=['full','drive'],default='full')
    p.add_argument('--freeze-gain',action='store_true')
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--gae-lambda',type=float,default=.95)
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--eval-episodes',type=int,default=10)
    a=p.parse_args();assert a.envs in [4,8,16] and a.sequences in [4,8,16]
    assert a.budget>0 and a.budget%a.envs==0 and a.budget<=500000
    assert a.memory>=1 and a.eval_episodes>=1 and 0<a.gamma<=1 and 0<=a.gae_lambda<=1 and a.entropy_coef>=0
    assert 1<=a.target<=150 and a.stall_frames>=0 and a.repeat>=1 and 0<=a.time_penalty<1
    if a.task=='bend':
        assert a.action_set=='drive' and a.repeat==8 and a.target==13 and a.stall_frames==600 and a.time_penalty==0
    train(a)
