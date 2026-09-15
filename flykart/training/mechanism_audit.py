"""Input-metadata audit and paired-history washout, without parameter updates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import supervised as base
from . import layer_probes as lp
OUT=lp.OUT.parent/'architecture_mechanisms_v1'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def prepare():
    OUT.mkdir(exist_ok=False)
    root=base.ROOT/'data/malecns'
    with np.load(root/'traced_smoke_graph.npz') as d:ids=d['body_ids'];visual=d['visual_mask'].astype(bool);rows=d['rows'];cols=d['cols'];values=d['values']
    ap=root/'body-annotations-male-cns-v1.0-minconf-0.5.feather';npth=root/'body-neurotransmitters-male-cns-v1.0.feather'
    a=pd.read_feather(ap).set_index('bodyId').loc[ids];nt=pd.read_feather(npth).set_index('body').reindex(ids)
    hexmask=a[['assignedOlHex1','assignedOlHex2']].notna().all(axis=1).to_numpy()&visual
    # Conservative sign hypothesis: high-confidence predicted GABA agreeing with consensus only.
    inhibitory=((nt.predicted_nt=='gaba')&(nt.consensus_nt=='gaba')&(nt.predicted_nt_confidence>=.8)).fillna(False).to_numpy()
    signs=np.where(inhibitory,-1.,1.).astype(np.float32)
    groups=(a.superclass.fillna('unknown').astype(str)+'|'+a.somaSide.fillna('unknown').astype(str)).to_numpy()
    rng=np.random.default_rng(20260915);shuffled=signs.copy()
    for g in np.unique(groups):idx=np.flatnonzero(groups==g);shuffled[idx]=rng.permutation(signs[idx])
    hids=np.flatnonzero(hexmask);xy=a.loc[hexmask,['assignedOlHex1','assignedOlHex2']].to_numpy().astype(np.float32)
    # Coordinate ordering only: no claim that axial coordinates are calibrated retinal azimuth/elevation.
    grid=xy.copy()
    for side in np.unique(a.somaSide.fillna('unknown').to_numpy()[hids]):
        ix=np.flatnonzero(a.somaSide.fillna('unknown').to_numpy()[hids]==side);z=xy[ix];grid[ix]=2*(z-z.min(0))/(z.max(0)-z.min(0))-1
    keys=(a.type.fillna('unknown').astype(str)+'|'+a.somaSide.fillna('unknown').astype(str)).to_numpy()[hids];perm=np.arange(len(hids))
    for g in np.unique(keys):ix=np.flatnonzero(keys==g);perm[ix]=rng.permutation(ix)
    np.savez(OUT/'metadata.npz',hex_ids=hids,grid=grid,shuffled_grid=grid[perm],grid_permutation=perm,signs=signs,shuffled_signs=shuffled)
    def counts(s):return {str(k):int(v) for k,v in s.fillna('missing').value_counts().items()}
    report=dict(nodes=len(ids),visual_nodes=int(visual.sum()),hex_nodes=len(hids),hex_fraction=float(hexmask.sum()/visual.sum()),hex_by_side=counts(a.loc[hexmask,'somaSide']),hex_shuffle_changed_fraction=float(np.any(grid!=grid[perm],axis=1).mean()),nt_consensus=counts(nt.consensus_nt),nt_missing_rows=int(nt.total_nt_predictions.isna().sum()),confidence_quantiles=nt.predicted_nt_confidence.quantile([0,.25,.5,.75,1]).to_dict(),high_confidence_consensus_gaba_nodes=int(inhibitory.sum()),inhibitory_outgoing_edges=int(inhibitory[cols].sum()),inhibitory_absolute_weight_fraction=float(values[inhibitory[cols]].sum()/values.sum()),sign_policy='Only predicted GABA == consensus GABA and body confidence >= .8 is negative. All others preserve baseline positive signs. Conservative hypothesis, not a complete biological E/I model.',sign_shuffle='within superclass x somaSide; neuron-level outgoing signs',spatial_policy='normalize two assignedOlHex coordinates per side to [-1,1]; bilinear 9x9 CNN sampling; shuffled coordinate pairs within type x somaSide; unassigned nodes retain old mapping',spatial_limit='Hex column labels are available, but no calibrated retinal screen transform or left/right orientation. Ordered artificial mapping only; 23% coverage.',sources={str(p):base.digest(p) for p in [ap,npth,root/'traced_smoke_graph.npz']})
    save('metadata_audit.json',report);print('METADATA',json.dumps(report),flush=True)

@torch.no_grad()
def decay():
    model,path=lp.backbone('k2');before=base.tensor_hash(model.state_dict());rows=base.dataset('validation')
    f=lp.Features(model);indices={'visual':f.indices['visual'],'intermediate':f.indices['intermediate'],'motor':model.readout};f.hook.remove()
    # Three trajectory pairs: different legal observed histories, then exactly identical subsequent inputs.
    results=[]
    variants=[('k1',1,.5,1.),('k2',2,.5,1.),('k4',4,.5,1.),('slower',2,.2,1.),('gain1.2',2,.5,1.2)]
    for label,steps,alpha,mult in variants:
        for pair in range(3):
            a=rows[2*pair]['obs'];b=rows[2*pair+1]['obs'];prefix=min(16,len(a),len(b));h=torch.zeros(model.n,2,device='cuda')
            gain=(.5+model.a.sigmoid())[:,None]*mult
            def update(obs,h):
                z=model.encoder(obs[:,None].float()/255);signal=z.T.index_select(0,model.channels)*model.visual[:,None]
                for _ in range(steps):h=(1-alpha)*h+alpha*torch.tanh(gain*torch.sparse.mm(model.w,h)+signal)
                return h
            for t in range(prefix):h=update(torch.stack([a[t],b[t]]),h)
            measures=[]
            for t in range(65):
                delta=h[:,0]-h[:,1];l=model.actor(h[model.readout].T);p=l.softmax(-1)
                measures.append(dict(decision=t,neural_updates=t*steps,global_linf=float(delta.abs().max()),rms={name:float(delta[ix].square().mean().sqrt()) for name,ix in indices.items()},logit_l2=float((l[0]-l[1]).norm()),probability_tv=float((p[0]-p[1]).abs().sum()/2),state_saturation_fraction=float((h.abs()>.99).float().mean())))
                if t<64:
                    o=a[min(prefix+t,len(a)-1)];h=update(o[None].expand(2,-1,-1),h)
            sums=torch.sparse.mm(model.w,torch.ones(model.n,1,device='cuda'));bound=(1-alpha)+alpha*float((gain*sums).max())
            assert all(v['global_linf']<=measures[i]['global_linf']*bound**steps+2e-6 for i,v in enumerate(measures[1:]))
            results.append(dict(variant=label,pair=pair,source_ids=[rows[2*pair]['meta']['start_id'],rows[2*pair+1]['meta']['start_id']],prefix_decisions=prefix,steps=steps,alpha=alpha,gain_multiplier=mult,neural_lipschitz_upper_bound=bound,decision_lipschitz_upper_bound=bound**steps,measurements=measures));print('DECAY',label,pair,flush=True)
    assert before==base.tensor_hash(model.state_dict());save('washout.json',dict(checkpoint=str(path),checkpoint_sha256=base.digest(path),backbone_unchanged=True,reset_policy='No artificial Context refresh during 64-decision measurement, to isolate neural dynamics.',caveat='Counterfactual variants are not retrained. Same future inputs erase prior-history differences; this does not measure continued visual injection quality or prove memory is needed.',results=results))
if __name__=='__main__':torch.set_num_threads(4);prepare();decay()
