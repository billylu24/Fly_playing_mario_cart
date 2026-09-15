"""Evaluate only the eight 24-frame starts validated by drive-only oracle."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import evaluate_vision_ablation as evaluation
from training.bend_task import BendTask,STARTS
from training.model import GainPolicy
from training.task import ROOT

out=ROOT/'tests/bend_validation'
gates=json.loads((out/'drive_oracle_24.json').read_text())
valid=[e['start_id'] for e in gates if e['target_reached']]
evaluation.Task=BendTask
torch.set_num_threads(4);torch.manual_seed(0)
model=GainPolicy(actions=3).eval()
model.load_state_dict(torch.load(ROOT/'runs/gain_matched_v4/last.pt',weights_only=False)['model'])
report={'perturb_frames':24,'excluded':['f600_left'],'episodes':[]}
for start in valid:
    cfg={'start_id':start,'perturb_frames':24}
    for mode in ['normal','frozen_image','black_image']:
        for seed in [4100,4101]:
            report['episodes'].append({'policy':mode,**evaluation.rollout(model,cfg,64,seed,mode)})
    task=BendTask(**cfg)
    try:
        for mode in ['straight','left','right','random']:
            task.reset();rng=np.random.default_rng(4100)
            while True:
                action=int(rng.integers(3)) if mode=='random' else {'straight':0,'left':1,'right':2}[mode]
                m=task.step(action)
                if m['done']:
                    report['episodes'].append({'policy':mode,'seed':4100,**m['episode']});break
    finally:task.close()
    (out/'strong_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(start,'completed',flush=True)
