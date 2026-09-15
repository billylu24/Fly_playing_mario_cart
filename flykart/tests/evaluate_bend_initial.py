"""Same validation protocol for step zero, no updates."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training.task import ROOT
from training.model import GainPolicy
from training.train_gain import evaluate
torch.set_num_threads(4);torch.manual_seed(0)
run=ROOT/'runs/bend_gain_32k_v1';model=GainPolicy(actions=3).eval()
model.load_state_dict(torch.load(run/'initial.pt',weights_only=False)['model'])
result={}
for vision in ['normal','frozen']:
    episodes=[]
    for start in ['f570_center','f570_left','f570_right']:
        episodes.extend(evaluate(model,seed_base=5000,episodes=3,
            task_config=dict(kind='bend',start_id=start,perturb_frames=24,action_set='drive'),memory=64,vision=vision))
    result[vision]={'episodes':episodes,'target_rate':sum(e['target_reached'] for e in episodes)/len(episodes)}
(run/'initial_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v['target_rate'] for k,v in result.items()})
