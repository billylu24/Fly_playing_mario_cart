"""Validate bend starts with privileged controller, controls and pixel ablations."""
import json,math,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from training.bend_task import BendTask,STARTS
from training.task import ROOT
from training.model import GainPolicy

out=ROOT/'tests/bend_validation';out.mkdir(exist_ok=True)
points=np.array([(464,260),(454,224),(420,206),(350,170),(260,120),(165,70),(100,42),(64,67),
                 (40,113),(40,300),(42,340),(65,366),(100,360),(200,315),(245,302),(265,318),
                 (312,400),(345,438),(385,452),(423,440),(456,415),(466,370),(464,310)],float)*8

def oracle(task,waypoint):
    info=task.env.data.lookup_all()
    pos=np.array([info['kart1_X'],info['kart1_Y']],float)
    if np.linalg.norm(points[waypoint]-pos)<220:waypoint=(waypoint+1)%len(points)
    delta=points[waypoint]-pos
    error=(math.atan2(delta[0],-delta[1])-info['kart1_direction']*2*math.pi/256+math.pi)%(2*math.pi)-math.pi
    action=4 if info['kart1_speed']>450 else 1
    if abs(error)>.12:action+=2 if error>0 else 1
    return action,waypoint

torch.set_num_threads(4);torch.manual_seed(0)
report={'starts':[],'episodes':[],'checkpoint':'runs/gain_matched_v4/last.pt'}
for spec in STARTS:
    task=BendTask(spec['id'],action_set='full')
    try:
        first=task.reset();info=task.start_info
        start={'spec':spec,'info':{k:info[k] for k in ['lap','current_checkpoint','kart1_X','kart1_Y','kart1_direction','kart1_speed','surface']},
               'pixels_sha256':hashlib.sha256(first.tobytes()).hexdigest()}
        cv2.imwrite(str(out/(spec['id']+'.png')),cv2.cvtColor(task.env.render(),cv2.COLOR_RGB2BGR))
        np.testing.assert_array_equal(first,task.reset());start['reset_verified']=True
        report['starts'].append(start)
        for policy in ['oracle','straight','left','right','random']:
            task.reset();rng=np.random.default_rng(4000);waypoint=5
            while True:
                if policy=='oracle':action,waypoint=oracle(task,waypoint)
                elif policy=='random':action=int(rng.integers(1,4))
                else:action={'straight':1,'left':2,'right':3}[policy]
                m=task.step(action)
                if m['done']:
                    report['episodes'].append({'policy':policy,'seed':4000,**m['episode']})
                    print(spec['id'],policy,m['episode']['reason'],m['episode']['advance'],flush=True);break
    finally:task.close()
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(e['target_reached'] for e in report['episodes'] if e['policy']=='oracle'),'Some starts are not oracle-validated'
model=GainPolicy(actions=3).eval()
model.load_state_dict(torch.load(ROOT/report['checkpoint'],weights_only=False)['model'])
for spec in STARTS:
    for mode in ['normal','frozen_image','black_image']:
        task=BendTask(spec['id'])
        try:
            for seed in range(4000,4003):
                obs=task.reset();initial=obs.copy();rng=torch.Generator().manual_seed(seed)
                h=torch.zeros(model.n,1,device='cuda');starts=torch.ones(1,device='cuda');history=[];step=0
                with torch.no_grad():
                    while True:
                        frame=initial if mode=='frozen_image' else np.zeros_like(obs) if mode=='black_image' else obs
                        o=torch.as_tensor(frame[None],device='cuda')
                        if step%32==0:
                            h=torch.zeros_like(h)
                            for old,mask in history[-64:]:_,_,h=model(old,h,mask)
                        history.append((o,starts.clone()));history=history[-64:]
                        logits,_,h=model(o,h,starts)
                        action=int(torch.multinomial(logits.softmax(-1)[0].cpu(),1,generator=rng))
                        m=task.step(action);obs=m['frame'];starts.fill_(0);step+=1
                        if m['done']:
                            report['episodes'].append({'policy':mode,'seed':seed,**m['episode']});break
            print(spec['id'],mode,'completed',flush=True)
        finally:task.close()
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPLETE',flush=True)
