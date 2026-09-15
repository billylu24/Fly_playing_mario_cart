"""Frozen k2 graph: matched raw/standardized motor readout and archived MLP."""
import copy,json
from pathlib import Path
import torch
from . import supervised as base
from . import layer_probes as lp
OUT=lp.OUT.parent/'readout_conditioning_v1'
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def run():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=False)
    save('protocol.json',dict(backbone='k2 seed0 best checkpoint, frozen',candidates=['raw_motor_linear','standard_motor_linear','standard_motor256_mlp32'],budget='raw: 600 full-batch Adam lr .003, validation every25; standardized probes reused from layer_probes_v1, identical training budget and seed123',selection='each checkpoint by validation normal BA, tie CE; evaluate all three prespecified heads, no closed-loop selection',controls='same archived time probe for all candidates; frozen first image for entire history; same 17 test starts and greedy/8100/8101/8102 sampling',limits='Exploratory reused test set, one backbone seed; frozen readout experiment does not test end-to-end normalization',source_sha256=base.digest(Path(__file__))))
    m,path=lp.backbone('k2');before=base.tensor_hash(m.state_dict());hashes={str(path):base.digest(path)};f=lp.Features(m)
    rows={s:base.dataset(s) for s in ['train','validation','test']};data={}
    for s in ['train','validation','test']:
        for mode in ['normal','frozen']:
            data[s,mode]=lp.extract(f,rows[s],mode);print('EXTRACTED',s,mode,flush=True)
    x,y,_=data['train','normal'];v,vy,_=data['validation','normal'];sl=f.slices['motor'];x=x[:,sl];v=v[:,sl]
    torch.manual_seed(123);raw=lp.Probe(x,'linear');raw.mean.zero_();raw.scale.fill_(1)
    initial=copy.deepcopy(raw.net.state_dict());opt=torch.optim.Adam(raw.parameters(),lr=.003);weights=len(y)/(3*torch.bincount(y,minlength=3).float());history=[];best=None
    for step in range(601):
        if step%25==0:
            score=lp.score(raw,v,vy);history.append(dict(step=step,**score));key=(score['balanced_accuracy'],-score['cross_entropy'])
            if best is None or key>best[0]:best=(key,copy.deepcopy(raw.state_dict()),step)
        if step==600:break
        opt.zero_grad();loss=torch.nn.functional.cross_entropy(raw(x),y,weight=weights);loss.backward();opt.step()
    raw.load_state_dict(best[1]);raw.requires_grad_(False);torch.save(raw.state_dict(),OUT/'raw_motor_linear.pt')
    archive=json.loads((lp.OUT/'k2_results.json').read_text());heads={'raw_motor_linear':('motor',raw)};reports={'raw_motor_linear':dict(selected_step=best[2],history=history)}
    for name,layer,typ in [('standard_motor_linear','motor','linear'),('standard_motor256_mlp32','motor256','mlp32')]:
        torch.manual_seed(123);p=lp.Probe(data['train','normal'][0][:,f.slices[layer]],typ)
        if typ=='linear':
            for key,tensor in initial.items():torch.testing.assert_close(tensor,p.net.state_dict()[key],rtol=0,atol=0)
        file=lp.OUT/f'k2_{layer}_{typ}.pt';hashes[str(file)]=base.digest(file);p.load_state_dict(torch.load(file,weights_only=True));p.requires_grad_(False);heads[name]=(layer,p);reports[name]=dict(reused_checkpoint=str(file),selected_step=archive[f'{layer}_{typ}']['selected_step'])
    checks={}
    for name,(layer,p) in heads.items():
        r=reports[name];r['offline']={s:{mode:lp.score(p,data[s,mode][0][:,f.slices[layer]],data[s,mode][1]) for mode in ['normal','frozen']} for s in rows}
        if name.startswith('standard'):
            source=archive[layer+('_linear' if name.endswith('linear') else '_mlp32')]
            for s in rows:
                for mode in ['normal','frozen']:assert r['offline'][s][mode]['confusion_matrix']==source[s][mode]['confusion_matrix']
        policy=lp.Policy(f,layer,p)
        with torch.no_grad():
            ctx=base.Context(policy);actual=torch.stack([ctx.step(o) for o in rows['validation'][0]['obs']]);expected=p(data['validation','normal'][0][:len(actual),f.slices[layer]])
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-5)
        checks[name]=dict(deployment_matches_extraction=True)
        print('OFFLINE',name,r['offline']['test'],flush=True)
    save('offline.json',reports)
    sd=x.std(0);save('feature_scales.json',dict(motor_std_quantiles=torch.quantile(sd,torch.tensor([0.,.25,.5,.75,1.],device='cuda')).tolist(),below_floor=int((sd<1e-5).sum()),dimension=x.shape[1],linear_initial_weights_identical=True))
    timefile=lp.OUT/'time_probe.pt';hashes[str(timefile)]=base.digest(timefile);time=lp.Probe(torch.zeros(2,1,device='cuda'),'linear');time.load_state_dict(torch.load(timefile,weights_only=True))
    with torch.no_grad():clock=list(time(torch.arange(225,device='cuda')[:,None]/225.).softmax(-1).cpu())
    pool=base.EvaluationPool()
    try:
        for name,(layer,p) in heads.items():
            policy=lp.Policy(f,layer,p).eval();report=dict(candidate=name,episodes=[])
            for at,row in enumerate(rows['test']):
                episodes=pool.rollout(policy,row['meta']['config'],clock)
                if at==0:
                    ref=base.closed_loop(policy,row['meta']['config'],'normal',8100,clock);actual=next(e for e in episodes if e['mode']=='normal' and e['sample_seed']==8100)
                    for key in ['actions','reason','frames','progress','jumps']:assert ref[key]==actual[key],key
                    checks[name]['batched_sequential_match']=True
                report['episodes'].extend(episodes);save(name+'_evaluation.json',report);print('EVALUATED',name,at+1,flush=True)
    finally:pool.close()
    assert before==base.tensor_hash(m.state_dict());assert all(p.grad is None for p in m.parameters());assert all(base.digest(Path(p))==h for p,h in hashes.items())
    save('checks.json',checks);save('status.json',dict(state='completed',backbone_unchanged=True,no_backbone_gradients=True,source_checkpoints_unchanged=True,source_hashes=hashes));f.hook.remove()
if __name__=='__main__':run()
