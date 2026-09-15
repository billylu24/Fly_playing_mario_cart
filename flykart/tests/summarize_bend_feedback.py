"""Aggregate the predeclared bend feedback experiment without model selection."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'tests/bend_feedback_v1'
g=json.loads((out/'gates.json').read_text());r=json.loads((out/'models.json').read_text())
assert r.get('checkpoint_hashes_unchanged') and len(r['episodes'])==len(g['selected'])*5*4
ids=[c['start_id'] for c in g['selected']]
summary={'diagnostics':{},'by_start':{},'paired':{},'controls':{},'checks':{'normal_evaluator_regression_passed':r['normal_evaluator_regression_passed'],'checkpoint_hashes_unchanged':r['checkpoint_hashes_unchanged']}}
keys=['pixel_frozen_mae','encoder_temporal_std','encoder_normal_rms','encoder_frozen_rms_diff','encoder_frozen_relative_rms','readout_normal_rms','readout_frozen_rms_diff','readout_frozen_relative_rms','frozen_action_tv_mean','black_action_tv_mean','encoder_saturated_fraction']
for ck in r['checkpoints']:
 es=[e for e in r['diagnostics'] if e['checkpoint']==ck]
 summary['diagnostics'][ck]={k:mean(e[k] for e in es) for k in keys}
for mode in ['normal','frozen','midfreeze','clock']:
 rows=[e for e in r['episodes'] if e['policy']==mode]
 summary[mode]={'passed':sum(e['target_reached'] for e in rows),'episodes':len(rows),'mean_advance':mean(e['advance'] for e in rows),'mean_frames':mean(e['frames'] for e in rows)}
for ident in ids:
 summary['by_start'][ident]={mode:sum(e['target_reached'] for e in r['episodes'] if e['start_id']==ident and e['policy']==mode) for mode in ['normal','frozen','midfreeze','clock']}
normal={(e['start_id'],e['seed']):e for e in r['episodes'] if e['policy']=='normal'}
for mode in ['frozen','midfreeze','clock']:
 pairs=[(normal[(e['start_id'],e['seed'])],e) for e in r['episodes'] if e['policy']==mode]
 summary['paired'][mode]={'same_action_sequence':sum(a['actions']==b['actions'] for a,b in pairs),'same_outcome':sum(all(a[k]==b[k] for k in ['frames','reason','advance','target_reached']) for a,b in pairs),'changed_success':sum(a['target_reached']!=b['target_reached'] for a,b in pairs),'normal_only_success':sum(a['target_reached'] and not b['target_reached'] for a,b in pairs),'intervention_only_success':sum(b['target_reached'] and not a['target_reached'] for a,b in pairs)}
 if mode=='midfreeze':summary['paired'][mode]['normal_episodes_reaching_freeze']=sum(len(a['actions'])>16 for a,b in pairs)
for mode in ['oracle','straight','left','right','sequence']:
 rows=[e for e in g['controls'] if e['start_id'] in ids and e['policy']==mode]
 summary['controls'][mode]={'passed':sum(e['target_reached'] for e in rows),'episodes':len(rows)}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
