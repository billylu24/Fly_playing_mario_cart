"""Final checkpoints: gradients along allowed visual inputs, and derivative checks."""
import json
import numpy as np
import torch
from . import layer_probes as lp
from . import supervised as base
from .transport_reconstruction import OUT,Transport,KINDS,save

def run():
    torch.set_num_threads(3);assert (OUT/'status.json').exists();data=torch.load(OUT/'features.pt',weights_only=False);z=data['z'].cuda();stats={k:v.cuda() for k,v in data['stats'].items()};target=(z-stats['zmean'])/stats['zscale']
    with np.load(OUT/'edges.npz') as d:edges=tuple(torch.tensor(d[k],device='cuda') for k in ['src','dst','weights'])
    protocol=json.loads((OUT/'protocol.json').read_text());training=base.dataset('train');validation=base.dataset('validation')
    train_rows=[next(r for r in training if r['meta']['start_id']==i) for i in protocol['training_ids']]
    labels={'train':torch.stack([r['labels'][:24] for r in train_rows],1).flatten(),'validation':torch.stack([r['labels'][:21] for r in validation],1).flatten()}
    head=lp.Probe(torch.zeros(2,64,device='cuda'),'mlp32');head.load_state_dict(torch.load(lp.OUT/'k2_cnn_mlp32.pt',weights_only=True));head.requires_grad_(False)
    results={}
    for kind in KINDS:
        b,_=lp.backbone('k2');m=Transport(b,kind,stats,edges);m.load_state_dict(torch.load(OUT/(kind+'.pt'),weights_only=True));m.zero_grad(set_to_none=True);zz=z.clone().requires_grad_(True);pred,states,signals=m.sequence(zz,True);loss=(pred[-1]-target[-1]).square().mean();loss.backward();visual=b.visual.bool()
        record=dict(visual_signal_gradient_rms=[float(s.grad[visual].square().mean().sqrt()) for s in signals],latent_input_gradient_rms=[float(g.square().mean().sqrt()) for g in zz.grad],note='Only permitted visual injection coordinates; latent gradients include channel mapping and interface. Targets detached, no gradient through teacher.',finite_differences={})
        # Check implemented parameter derivatives on one aggregate normalized direction per newly opened module.
        m.zero_grad(set_to_none=True);pred,_,_=m.sequence(z);full=(pred-target).square().mean();full.backward()
        for name in ['interface','delta']:
            p=getattr(m,name)
            if not p.requires_grad:continue
            norm=p.grad.norm();direction=p.grad/norm.clamp_min(1e-30);original=p.detach().clone();eps=.001
            with torch.no_grad():
                p.copy_(original+eps*direction);plus=(m.sequence(z)[0]-target).square().mean();p.copy_(original-eps*direction);minus=(m.sequence(z)[0]-target).square().mean();p.copy_(original)
            observed=float((plus-minus)/(2*eps));expected=float(norm);relative=abs(observed-expected)/max(abs(expected),1e-8)
            assert relative<.1,(kind,name,relative,observed,expected)
            record['finite_differences'][name]=dict(autograd_directional_derivative=expected,finite_difference=observed,relative_error=relative)
        record['downstream_actions']={}
        for split,latents in [('train',z),('validation',data['validation_z'].cuda())]:
            with torch.no_grad():
                rebuilt=m.sequence(latents)[0]*stats['zscale']+stats['zmean'];actual=head(rebuilt.flatten(0,1));reference=head(latents.flatten(0,1))
            record['downstream_actions'][split]=dict(reconstructed=base.scores(labels[split].cpu().numpy(),actual.cpu().numpy()),direct_cnn=base.scores(labels[split].cpu().numpy(),reference.cpu().numpy()),agreement_with_direct_cnn=float((actual.argmax(-1)==reference.argmax(-1)).float().mean()))
        results[kind]=record;del m,b
    save('allowed_gradient_audit.json',results);print('ALLOWED GRADIENT / DERIVATIVE CHECKS PASS',flush=True)
if __name__=='__main__':run()
