"""Bounded real-ROM, real-connectome recurrent PPO smoke test, not a training run.

Four environments × 32 decisions × four updates, two PPO epochs/update.
Uses integration's existing reward/done only; never claims finish detection.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
ACTIONS=[[],['B'],['B','LEFT'],['B','RIGHT'],['Y'],['Y','LEFT'],['Y','RIGHT']]


def worker(conn):
    os.environ.pop('DISPLAY',None);os.environ.pop('WAYLAND_DISPLAY',None)
    os.environ['SDL_AUDIODRIVER']='dummy'
    import cv2
    import numpy as np
    import stable_retro as retro
    cv2.setNumThreads(1)
    retro.data.Integrations.add_custom_path(str(ROOT/'mario_kart'))
    env=None
    def frame(obs):
        return cv2.resize(cv2.cvtColor(obs,cv2.COLOR_RGB2GRAY),(84,84),interpolation=cv2.INTER_AREA)
    try:
        env=retro.make('SuperMarioKart-Snes',inttype=retro.data.Integrations.CUSTOM_ONLY,render_mode='rgb_array')
        action_vectors=[]
        for names in ACTIONS:
            v=np.zeros(len(env.buttons),dtype=np.int8)
            for name in names:v[env.buttons.index(name)]=1
            action_vectors.append(v)
        obs,_=env.reset();conn.send({'frame':frame(obs)})
        while True:
            cmd=conn.recv()
            if cmd=='close':break
            reward=0.;done=False
            for frames in range(1,5):
                obs,r,terminated,truncated,info=env.step(action_vectors[int(cmd)])
                # Native integration is terminal-only; do not silently mishandle time limits.
                assert not truncated
                reward+=float(r);done=terminated
                if done:break
            terminal_info={k:info.get(k) for k in ['lap','current_checkpoint','kart1_speed','surface','getGameMode']}
            if done:obs,_=env.reset()
            conn.send({'frame':frame(obs),'reward':reward,'done':done,'frames':frames,'info':terminal_info})
    except Exception as e:
        conn.send({'error':repr(e)})
    finally:
        if env is not None:env.close()
        conn.close()


def main(updates,steps,batch,output):
    import numpy as np
    import torch
    from torch import nn
    from torch.distributions import Categorical
    from sparse_edge_ops import Graph,multiply
    torch.manual_seed(42);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.55)
    t0=time.perf_counter()
    graph_data=np.load(ROOT/'data/malecns/traced_smoke_graph.npz')
    n=len(graph_data['body_ids']);e=len(graph_data['values'])
    indices=torch.stack([torch.as_tensor(graph_data['rows'],device='cuda',dtype=torch.int64),
                         torch.as_tensor(graph_data['cols'],device='cuda',dtype=torch.int64)])
    graph=Graph(indices,n)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.edges=nn.Parameter(torch.as_tensor(graph_data['values'],device='cuda'))
            self.encoder=nn.Sequential(nn.Conv2d(1,16,8,4),nn.ReLU(),nn.Conv2d(16,32,4,2),nn.ReLU(),
                                       nn.Flatten(),nn.Linear(32*9*9,64),nn.Tanh()).cuda()
            self.register_buffer('channels',torch.arange(n,device='cuda')%64)
            self.register_buffer('visual',torch.as_tensor(graph_data['visual_mask'],device='cuda',dtype=torch.float32))
            self.register_buffer('readout',torch.as_tensor(graph_data['readout_ids'],device='cuda'))
            self.actor=nn.Linear(len(self.readout),len(ACTIONS),device='cuda')
            self.critic=nn.Linear(len(self.readout),1,device='cuda')
        def forward(self,obs,h,starts):
            h=h*(1-starts[None,:])
            z=self.encoder(obs[:,None].float()/255)
            injection=z.T.index_select(0,self.channels)*self.visual[:,None]
            h=0.5*h+0.5*torch.tanh(multiply('edge_triton',self.edges,h,graph)+injection)
            read=h.index_select(0,self.readout).T
            return self.actor(read),self.critic(read).squeeze(-1),h

    policy=Policy();graph_data.close()
    opt=torch.optim.Adam(policy.parameters(),lr=1e-4,eps=1e-5,foreach=False)
    torch.cuda.synchronize()
    setup_seconds=time.perf_counter()-t0
    allocated_after_graph=torch.cuda.memory_allocated()/2**20
    ctx=mp.get_context('spawn');parents=[];children=[]
    report={'status':'running','nodes':n,'edges':e,'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'batch':batch,'unroll':steps,'updates':updates,'ppo_epochs':2,
            'action_repeat':4,'actions':ACTIONS,'dtype':'float32','seed':42,
            'learning_rate':1e-4,'gamma':0.99,'gae_lambda':0.95,'clip':0.2,
            'value_coefficient':0.5,'entropy_coefficient':0.01,'max_grad_norm':0.5,
            'graph_setup_seconds':setup_seconds,'allocated_after_graph_mib':allocated_after_graph,
            'trainable_parameters':sum(p.numel() for p in policy.parameters()),'iterations':[]}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    def receive(conn):
        if not conn.poll(30):raise TimeoutError('emulator did not reply within 30 seconds')
        msg=conn.recv()
        if 'error' in msg:raise RuntimeError(msg['error'])
        return msg
    try:
        for _ in range(batch):
            parent,child=ctx.Pipe();proc=ctx.Process(target=worker,args=(child,));proc.start();child.close()
            parents.append(parent);children.append(proc)
        obs=torch.as_tensor(np.stack([receive(p)['frame'] for p in parents]),device='cuda')
        h=torch.zeros(n,batch,device='cuda');starts=torch.ones(batch,device='cuda')
        all_frames=0
        for update in range(updates):
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();iteration_start=time.perf_counter()
            initial_h=h.detach().clone()
            obs_buf=[];start_buf=[];action_buf=[];logp_buf=[];value_buf=[];reward_buf=[];done_buf=[]
            resets=0;frames_count=0;env_seconds=0.;policy_seconds=0.;infos=[]
            for _ in range(steps):
                tick=time.perf_counter()
                with torch.no_grad():
                    logits,v,h=policy(obs,h,starts)
                    distribution=Categorical(logits=logits);actions=distribution.sample();logp=distribution.log_prob(actions)
                action_list=actions.cpu().tolist();torch.cuda.synchronize();policy_seconds+=time.perf_counter()-tick
                obs_buf.append(obs);start_buf.append(starts);action_buf.append(actions);logp_buf.append(logp);value_buf.append(v)
                tick=time.perf_counter()
                for p,a in zip(parents,action_list):p.send(a)
                msgs=[receive(p) for p in parents]
                obs=torch.as_tensor(np.stack([m['frame'] for m in msgs]),device='cuda')
                reward_buf.append(torch.tensor([m['reward'] for m in msgs],device='cuda'))
                starts=torch.tensor([float(m['done']) for m in msgs],device='cuda');done_buf.append(starts)
                resets+=int(starts.sum().item());frames_count+=sum(m['frames'] for m in msgs)
                infos=[m['info'] for m in msgs];torch.cuda.synchronize();env_seconds+=time.perf_counter()-tick
            with torch.no_grad():
                _,next_value,_=policy(obs,h,starts)
            observations=torch.stack(obs_buf);episode_starts=torch.stack(start_buf)
            actions=torch.stack(action_buf);old_logp=torch.stack(logp_buf);old_values=torch.stack(value_buf)
            rewards=torch.stack(reward_buf);dones=torch.stack(done_buf)
            advantages=torch.zeros_like(rewards);carry=torch.zeros(batch,device='cuda')
            for t in reversed(range(steps)):
                nv=next_value if t==steps-1 else old_values[t+1]
                delta=rewards[t]+0.99*nv*(1-dones[t])-old_values[t]
                carry=delta+0.99*0.95*(1-dones[t])*carry;advantages[t]=carry
            returns=advantages+old_values
            advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
            torch.cuda.synchronize();rollout_end=time.perf_counter()
            epoch_records=[]
            for epoch in range(2):
                opt.zero_grad(set_to_none=True)
                unrolled_h=initial_h
                logps=[];vals=[];entropies=[]
                tick=time.perf_counter()
                for t in range(steps):
                    logits,v,unrolled_h=policy(observations[t],unrolled_h,episode_starts[t])
                    dist=Categorical(logits=logits)
                    logps.append(dist.log_prob(actions[t]));vals.append(v);entropies.append(dist.entropy())
                new_logp=torch.stack(logps);values=torch.stack(vals);entropy=torch.stack(entropies).mean()
                if epoch==0:
                    replay_error=(new_logp-old_logp).abs().max().item()
                    assert replay_error<2e-5,f'recurrent replay mismatch: {replay_error}'
                ratio=(new_logp-old_logp).exp()
                actor_loss=-torch.minimum(ratio*advantages,ratio.clamp(0.8,1.2)*advantages).mean()
                value_loss=0.5*(values-returns).square().mean()
                loss=actor_loss+0.5*value_loss-0.01*entropy
                assert torch.isfinite(loss).item()
                torch.cuda.synchronize();fwd_end=time.perf_counter()
                loss.backward();torch.cuda.synchronize();bwd_end=time.perf_counter()
                groups={'edges':policy.edges,'encoder':policy.encoder[0].weight,
                        'actor':policy.actor.weight,'critic':policy.critic.weight}
                grad_norms={k:p.grad.norm().item() if p.grad is not None else None for k,p in groups.items()}
                assert all(v is not None and np.isfinite(v) and v>0 for v in grad_norms.values()),grad_norms
                total_norm=nn.utils.clip_grad_norm_(policy.parameters(),0.5,error_if_nonfinite=True).item()
                # Track an edge with largest gradient to confirm actual Adam mutation.
                tracked_index=int(policy.edges.grad.abs().argmax().item())
                before=policy.edges[tracked_index].item()
                opt.step();torch.cuda.synchronize();end=time.perf_counter()
                edge_delta=abs(policy.edges[tracked_index].item()-before)
                assert edge_delta>0,'edge parameter did not update'
                epoch_records.append({'forward_loss_ms':(fwd_end-tick)*1000,'backward_ms':(bwd_end-fwd_end)*1000,
                                      'optimizer_checks_ms':(end-bwd_end)*1000,'total_ms':(end-tick)*1000,
                                      'loss':loss.item(),'actor_loss':actor_loss.item(),'value_loss':value_loss.item(),
                                      'entropy':entropy.item(),'gradient_norms':grad_norms,'grad_norm_before_clip':total_norm,
                                      'tracked_edge_change':edge_delta,'replay_error_first_epoch':replay_error,
                                      'approx_kl':((ratio-1)-(new_logp-old_logp)).mean().item(),
                                      'clip_fraction':((ratio-1).abs()>0.2).float().mean().item()})
                del unrolled_h,logps,vals,entropies,dist,logits,v,new_logp,values,entropy,ratio,actor_loss,value_loss,loss
            h=h.detach();torch.cuda.synchronize();iteration_end=time.perf_counter();all_frames+=frames_count
            record={'update':update,'warmup':update==0,'rollout_seconds':rollout_end-iteration_start,
                    'policy_inference_seconds':policy_seconds,'env_ipc_preprocessing_seconds':env_seconds,
                    'update_seconds':iteration_end-rollout_end,'total_seconds':iteration_end-iteration_start,
                    'decisions':steps*batch,'emulator_frames':frames_count,'resets':resets,
                    'reward_sum':rewards.sum().item(),'reward_min':rewards.min().item(),'reward_max':rewards.max().item(),
                    'reward_std':rewards.std(unbiased=False).item(),'last_game_info':infos,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'epochs':epoch_records}
            report['iterations'].append(record);output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k not in ['epochs','last_game_info']}),flush=True)
        report['status']='passed';report['total_emulator_frames']=all_frames
        report['total_decisions']=updates*steps*batch
        report['wall_seconds_including_setup']=time.perf_counter()-t0
        output.write_text(json.dumps(report,indent=2)+'\n')
    except Exception as e:
        report['status']='failed';report['error']=repr(e)
        output.write_text(json.dumps(report,indent=2)+'\n');raise
    finally:
        for p in parents:
            try:p.send('close')
            except (BrokenPipeError,EOFError):pass
        for proc in children:
            proc.join(timeout=5)
            if proc.is_alive():proc.terminate();proc.join(timeout=5)
        for p in parents:p.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=4)
    parser.add_argument('--steps',type=int,default=32);parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--output',default=str(ROOT/'tests/rl_smoke_runs/default.json'))
    a=parser.parse_args()
    assert 1<=a.updates<=8 and 1<=a.steps<=64 and 1<=a.batch<=4,'bounded smoke test only'
    main(a.updates,a.steps,a.batch,a.output)
