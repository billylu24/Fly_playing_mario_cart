"""Summarize all predeclared paired seeds; no selection on test performance."""
import json,sys
from pathlib import Path
from statistics import mean,stdev
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.supervised import EXPERIMENT,digest


def main():
    summary=dict(data=json.loads((EXPERIMENT/'data_summary.json').read_text()),runs={},paired_initialization={},aggregates={})
    plan=json.loads((EXPERIMENT/'plan.json').read_text())
    for seed in plan['seeds']:
        configs={arch:json.loads((EXPERIMENT/f'{arch}_seed{seed}'/'config.json').read_text()) for arch in ['graph','gru']}
        assert configs['graph']['encoder_initial_sha256']==configs['gru']['encoder_initial_sha256']
        summary['paired_initialization'][str(seed)]=configs['graph']['encoder_initial_sha256']
        statuses={arch:json.loads((EXPERIMENT/f'{arch}_seed{seed}'/'status.json').read_text()) for arch in ['graph','gru']}
        assert statuses['graph']['valid_presentations']==statuses['gru']['valid_presentations']
        for arch in ['graph','gru']:
            run=EXPERIMENT/f'{arch}_seed{seed}';r=json.loads((run/'evaluation.json').read_text())
            assert r['checkpoint_unchanged'] and r['batched_evaluator_regression_passed']
            expected=summary['data']['test']['accepted']*12;assert len(r['episodes'])==expected
            result=dict(seed=seed,architecture=arch,selected_step=r['selected_step'],offline=r['offline'],status=statuses[arch],closed_loop={},by_start={})
            for mode in ['normal','frozen','clock']:
                result['closed_loop'][mode]={}
                for greedy in [True,False]:
                    rows=[e for e in r['episodes'] if e['mode']==mode and (e['sample_seed'] is None)==greedy]
                    result['closed_loop'][mode]['greedy' if greedy else 'sampled']=dict(passed=sum(e['target_reached'] for e in rows),episodes=len(rows),mean_advance=mean(e['advance'] for e in rows),mean_frames=mean(e['frames'] for e in rows))
            for ident in sorted({e['start_id'] for e in r['episodes']}):
                result['by_start'][ident]={mode:dict(greedy=next(e['target_reached'] for e in r['episodes'] if e['start_id']==ident and e['mode']==mode and e['sample_seed'] is None),sampled=sum(e['target_reached'] for e in r['episodes'] if e['start_id']==ident and e['mode']==mode and e['sample_seed'] is not None)) for mode in ['normal','frozen','clock']}
            summary['runs'][run.name]=result
    for arch in ['graph','gru']:
        rows=[r for r in summary['runs'].values() if r['architecture']==arch]
        aggregate={}
        for mode in ['normal','frozen']:
            a=[r['offline']['test'][mode]['balanced_accuracy'] for r in rows]
            aggregate[mode+'_test_balanced_accuracy']=dict(mean=mean(a),sample_std=stdev(a),per_seed=a)
        for mode in ['normal','frozen','clock']:
            for sampling in ['greedy','sampled']:
                a=[r['closed_loop'][mode][sampling] for r in rows]
                aggregate[mode+'_'+sampling]=dict(passed=sum(x['passed'] for x in a),episodes=sum(x['episodes'] for x in a),per_seed=[x['passed'] for x in a])
        summary['aggregates'][arch]=aggregate
    (EXPERIMENT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['aggregates'],indent=2))
if __name__=='__main__':main()
