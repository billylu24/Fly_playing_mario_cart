"""Matched-seed comparison of learned visual projection and archived baselines."""
import json,sys
from pathlib import Path
from statistics import mean,stdev
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.injection_experiment import OUTPUT,ORIGINAL
from training.supervised import digest

def main():
    plan=json.loads((OUTPUT/'plan.json').read_text())
    assert digest(OUTPUT/'groups.npz')==plan['groups_sha256']
    assert digest(Path(__file__).resolve().parents[1]/'data/malecns/traced_smoke_graph.npz')==plan['graph_sha256']
    summary=dict(runs={},checks=json.loads((OUTPUT/'checks.json').read_text()),baseline_source=str(ORIGINAL))
    for seed in [0,1,2]:
        path=OUTPUT/f'graph_seed{seed}';ref=ORIGINAL/f'graph_seed{seed}'
        for arch in ['graph','gru']:
            archived=ORIGINAL/f'{arch}_seed{seed}'
            oldeval=json.loads((archived/'evaluation.json').read_text())
            assert digest(archived/'best.pt')==oldeval['checkpoint_sha256']
        initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)['model']
        prior=torch.load(ref/'initial.pt',map_location='cpu',weights_only=False)['model']
        for k,v in prior.items():torch.testing.assert_close(v,initial[k],rtol=0,atol=0)
        assert torch.count_nonzero(initial['projection_delta'])==0
        r=json.loads((path/'evaluation.json').read_text());assert r['checkpoint_unchanged'] and r['batched_evaluator_regression_passed']
        assert len(r['episodes'])==204 and digest(path/'best.pt')==r['checkpoint_sha256']
        status=json.loads((path/'status.json').read_text());oldstatus=json.loads((ref/'status.json').read_text())
        assert status['fixed_w_unchanged'] and status['valid_presentations']==oldstatus['valid_presentations']
        result=dict(seed=seed,selected_step=r['selected_step'],offline=r['offline'],status=status,closed_loop={},by_start={},injection=json.loads((path/'injection_summary.json').read_text()))
        for mode in ['normal','frozen','clock']:
            result['closed_loop'][mode]={}
            for greedy in [True,False]:
                rows=[e for e in r['episodes'] if e['mode']==mode and (e['sample_seed'] is None)==greedy]
                result['closed_loop'][mode]['greedy' if greedy else 'sampled']=dict(passed=sum(e['target_reached'] for e in rows),episodes=len(rows),mean_advance=mean(e['advance'] for e in rows))
        for ident in sorted({e['start_id'] for e in r['episodes']}):
            result['by_start'][ident]={mode:dict(greedy=next(e['target_reached'] for e in r['episodes'] if e['start_id']==ident and e['mode']==mode and e['sample_seed'] is None),sampled=sum(e['target_reached'] for e in r['episodes'] if e['start_id']==ident and e['mode']==mode and e['sample_seed'] is not None)) for mode in ['normal','frozen','clock']}
        summary['runs'][str(seed)]=result
    aggregate={}
    for mode in ['normal','frozen']:
        values=[r['offline']['test'][mode]['balanced_accuracy'] for r in summary['runs'].values()]
        aggregate[mode+'_test_balanced_accuracy']=dict(mean=mean(values),sample_std=stdev(values),per_seed=values)
    for mode in ['normal','frozen','clock']:
        for sampling in ['greedy','sampled']:
            rows=[r['closed_loop'][mode][sampling] for r in summary['runs'].values()]
            aggregate[mode+'_'+sampling]=dict(passed=sum(r['passed'] for r in rows),episodes=sum(r['episodes'] for r in rows),per_seed=[r['passed'] for r in rows])
    summary['aggregates']=dict(injection=aggregate,**json.loads((ORIGINAL/'summary.json').read_text())['aggregates'])
    summary['shared_initial_tensors_all_seeds_identical']=True
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
