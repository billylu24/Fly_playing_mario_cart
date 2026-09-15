"""Ensure actual trained coordinate permutation changes injection and logits."""
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training import supervised as base
from training.mechanism_experiment import Policy,OUT
@torch.no_grad()
def main():
    torch.set_num_threads(2);m=Policy('spatial');m.load_state_dict(torch.load(OUT/'spatial/graph_seed0/best.pt',weights_only=False)['model']);obs=base.dataset('validation')[0]['obs'][:16];original=m.grid.clone()
    with np.load(OUT/'metadata.npz') as d:shuffled=torch.tensor(d['shuffled_grid'],device='cuda')[None,:,None,:]
    def rollout(grid):
        m.grid.copy_(grid);h=torch.zeros(m.n,1,device='cuda');ls=[]
        for i,o in enumerate(obs):l,_,h=m(o[None],h,torch.tensor([float(i==0)],device='cuda'));ls.append(l)
        return torch.cat(ls)
    normal=rollout(original);swapped=rollout(shuffled)
    conv=m.encoder[:4](obs[:,None].float()/255);ports=m.projection(conv)
    def sampled(grid):return torch.nn.functional.grid_sample(ports,grid.expand(len(obs),-1,-1,-1),align_corners=True).squeeze(-1).gather(1,m.channels[m.hex_ids][None,None,:].expand(len(obs),1,-1)).squeeze(1)
    delta=sampled(original)-sampled(shuffled)
    result=dict(injection_coordinate_swap_rms=float(delta.square().mean().sqrt()),injection_coordinate_swap_max=float(delta.abs().max()),logit_swap_rms=float((normal-swapped).square().mean().sqrt()),logit_swap_max=float((normal-swapped).abs().max()),sampled_observations=len(obs),identical_weights_used=True)
    assert result['injection_coordinate_swap_rms']>0 and result['logit_swap_max']>0
    (OUT/'spatial_intervention_checks.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
