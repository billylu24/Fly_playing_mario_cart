"""Read-only checkpoint rollout diagnosis. No backward or parameter updates."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import cv2
from training.model import GainPolicy
from training.task import Task,ROOT
from training.train_gain import gae

torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.55)
torch.manual_seed(0)
out=ROOT/'tests/gain_diagnosis';out.mkdir(exist_ok=True)
run=ROOT/'runs/gain_seed0_v1'
model=GainPolicy();reports=[]
for label,file in [('best','best_validation.pt'),('final','last.pt')]:
    checkpoint=torch.load(run/file,weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval()
    task=Task();obs=task.reset();h=torch.zeros(model.n,1,device='cuda');starts=torch.ones(1,device='cuda')
    rng=torch.Generator().manual_seed(1004)
    trace=[];rejections=[];prev_q=-1;actual_max=-1;near_zero=[];previous_frontier=0
    obs_samples=[];state_samples=[];prepolicy=[];tick=time.perf_counter()
    try:
        with torch.no_grad():
            for t in range(4500):
                o=torch.as_tensor(obs[None],device='cuda')
                if t%300==0:
                    obs_samples.append(o.clone());state_samples.append(h.clone());prepolicy.append(starts.clone())
                logits,v,h=model(o,h,starts);probs=logits.softmax(-1)[0].cpu()
                action=int(torch.multinomial(probs,1,generator=rng));msg=task.step(action)
                info=msg['info'];q=(info['lap']-128)*30+info['current_checkpoint'];actual_max=max(actual_max,q)
                frontier=msg['episode']['progress'] if msg['done'] else msg['progress']
                if q>previous_frontier+1 and q!=prev_q:
                    rejections.append({'decision':t,'previous_raw':prev_q,'raw':q,'frontier':frontier,**info})
                if t in [0,120,1000,3000,4499]:
                    cv2.imwrite(str(out/f'{label}_{t}.png'),cv2.cvtColor(task.env.render(),cv2.COLOR_RGB2BGR))
                trace.append({'decision':t,'action':action,'probabilities':probs.tolist(),'value':v.item(),
                              'reward':msg['reward'],'frontier':frontier,'raw_progress':q,**info})
                obs=msg['frame'];starts.fill_(float(msg['done']));prev_q=q;previous_frontier=frontier
                if msg['done']:break
            gains=.5+model.a.sigmoid()
            # Counterfactual on saved observations/states, without rerolling behavior.
            deltas={'gain_reset_to_one':[],'blank_current_image':[],'reset_hidden':[]}
            original_a=model.a.clone()
            for o,old_h,s in zip(obs_samples,state_samples,prepolicy):
                p=model(o,old_h,s)[0].softmax(-1)
                model.a.zero_();pg=model(o,old_h,s)[0].softmax(-1);model.a.copy_(original_a)
                pb=model(torch.zeros_like(o),old_h,s)[0].softmax(-1)
                ph=model(o,torch.zeros_like(old_h),s)[0].softmax(-1)
                for name,q in [('gain_reset_to_one',pg),('blank_current_image',pb),('reset_hidden',ph)]:
                    deltas[name].append(.5*(p-q).abs().sum().item())
            # GAE signal on fixed replay segments before normalization, gamma/lambda as train.
            adv_windows=[]
            for start in range(0,len(trace)-32,32):
                block=trace[start:start+32]
                reward=torch.tensor([[x['reward']] for x in block]);value=torch.tensor([[x['value']] for x in block])
                av,_=gae(reward,value,torch.zeros_like(reward),torch.tensor([trace[start+32]['value']]))
                adv_windows.append({'start':start,'has_progress_reward':any(x['reward']>0 for x in block),
                                    'adv_std':av.std(unbiased=False).item(),'adv_mean':av.mean().item()})
        speeds=np.array([x['kart1_speed'] for x in trace]);surfaces=np.array([x['surface'] for x in trace])
        probs=np.array([x['probabilities'] for x in trace])
        report={'label':label,'checkpoint_decisions':checkpoint['decisions'],'seed':1004,
                'episode':msg['episode'],'decisions':len(trace),'seconds':time.perf_counter()-tick,
                'raw_max_progress':actual_max,'action_frequency':(np.bincount([x['action'] for x in trace],minlength=7)/len(trace)).tolist(),
                'mean_action_probabilities':probs.mean(axis=0).tolist(),
                'mean_entropy':float(-(probs*np.log(probs+1e-12)).sum(axis=1).mean()),
                'speed_mean':float(speeds.mean()),'speed_zero_fraction':float((speeds==0).mean()),
                'wall_fraction':float((surfaces==128).mean()),'road_fraction':float((surfaces==64).mean()),
                'gain_abs_change_quantiles':torch.quantile((gains-1).abs(),torch.tensor([.5,.9,.99,1.],device='cuda')).tolist(),
                'immediate_action_total_variation':{k:{'mean':float(np.mean(v)),'max':float(max(v))} for k,v in deltas.items()},
                'rejected_raw_advances':rejections,'adv_windows':adv_windows}
        (out/f'{label}_trace.json').write_text(json.dumps(trace)+'\n');reports.append(report)
        (out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['adv_windows','rejected_raw_advances']}),flush=True)
    finally:task.close()
