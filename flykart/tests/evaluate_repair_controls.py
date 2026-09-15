"""Same-task controls, separate evaluation RNG, no learning."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from training.task import Task
from training.model import GainPolicy
from training.train_gain import evaluate

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--episodes',type=int,default=10)
a=p.parse_args();run=Path(a.run);config=json.loads((run/'config.json').read_text())
torch.set_num_threads(4);torch.manual_seed(0)
cfg=config['task_config'];results={}
for policy in ['random','straight','left','right']:
    task=Task(**cfg);episodes=[]
    try:
        for seed in range(2000,2000+a.episodes):
            rng=np.random.default_rng(seed);task.reset()
            while True:
                fixed={'straight':0,'left':1,'right':2}
                action=int(rng.integers(len(task.vectors))) if policy=='random' else fixed[policy]+(cfg['action_set']=='full')
                m=task.step(action)
                if m['done']:episodes.append({'seed':seed,**m['episode']});break
    finally:task.close()
    results[policy]=episodes
model=GainPolicy(actions=3 if cfg['action_set']=='drive' else 7,train_gain=not config['freeze_gain'])
for label in ['initial','last']:
    model.load_state_dict(torch.load(run/f'{label}.pt',weights_only=False)['model'])
    results[label]=evaluate(model,seed_base=2000,episodes=a.episodes,task_config=cfg,memory=config['memory'])
report={'task':cfg,'episodes':results,'summary':{k:{
    'mean_progress':float(np.mean([e['progress'] for e in v])),
    'target_rate':float(np.mean([e['target_reached'] for e in v])),
    'mean_frames':float(np.mean([e['frames'] for e in v]))} for k,v in results.items()}}
(run/'control_evaluations.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['summary'],indent=2))
