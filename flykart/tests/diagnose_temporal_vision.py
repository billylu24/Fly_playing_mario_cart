"""Read-only, fixed-action temporal input ablation; never updates checkpoints."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.model import GainPolicy
from training.task import Task, ROOT

torch.set_num_threads(4)
out = ROOT / 'tests/gain_diagnosis'
model = GainPolicy().eval()
reports = []
for label, filename in [('best', 'best_validation.pt'), ('final', 'last.pt')]:
    model.load_state_dict(torch.load(ROOT/'runs/gain_seed0_v1'/filename, weights_only=False)['model'])
    trace = json.loads((out/f'{label}_trace.json').read_text())
    task = Task()
    obs = task.reset()
    frozen = obs.copy()
    h = torch.zeros(model.n, 3, device='cuda')
    hg = torch.zeros(model.n, 1, device='cuda')
    original_a = model.a.detach().clone()
    records, features = [], []
    try:
        with torch.no_grad():
            for t in range(512):
                frames = torch.as_tensor(np.stack([obs, np.zeros_like(obs), frozen]), device='cuda')
                starts = torch.full((3,), float(t == 0), device='cuda')
                logits, _, h = model(frames, h, starts)
                p = logits.softmax(-1)
                z = model.encoder(frames[:, None].float()/255)
                features.append(z.cpu().numpy())
                model.a.zero_()
                lg, _, hg = model(frames[:1], hg, starts[:1])
                model.a.copy_(original_a)
                records.append({'step': t,
                    'blank_tv': float(.5*(p[0]-p[1]).abs().sum()),
                    'frozen_tv': float(.5*(p[0]-p[2]).abs().sum()),
                    'gain_reset_tv': float(.5*(p[0]-lg.softmax(-1)[0]).abs().sum())})
                obs = task.step(trace[t]['action'])['frame']
        f = np.stack(features)
        reports.append({'label': label, 'steps': 512,
            'visual_readout_overlap': int(model.visual[model.readout].sum()),
            'encoder_actual_saturated_fraction': float((np.abs(f[:,0])>.99).mean()),
            'encoder_actual_temporal_std_mean': float(f[:,0].std(axis=0).mean()),
            'encoder_actual_blank_abs_diff_mean': float(np.abs(f[:,0]-f[:,1]).mean()),
            'encoder_actual_frozen_abs_diff_mean': float(np.abs(f[:,0]-f[:,2]).mean()),
            'ablation_after_step_64': {key: {'mean': float(np.mean([r[key] for r in records[64:]])),
                                          'max': max(r[key] for r in records[64:])}
                                      for key in ['blank_tv', 'frozen_tv', 'gain_reset_tv']},
            'records': records})
        print(json.dumps({k:v for k,v in reports[-1].items() if k != 'records'}), flush=True)
    finally:
        task.close()
(out/'temporal_vision.json').write_text(json.dumps(reports, indent=2)+'\n')
