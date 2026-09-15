"""Report the fixed-budget pilot and predeclared validation-gated replications."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.microsteps_experiment import OUTPUT,ORIGINAL
from training.supervised import digest

def result(path):
    r=json.loads((path/'evaluation.json').read_text());assert r['checkpoint_unchanged'] and r['batched_evaluator_regression_passed'];assert len(r['episodes'])==204
    assert digest(path/'best.pt')==r['checkpoint_sha256']
    status=json.loads((path/'status.json').read_text());assert status['fixed_w_unchanged']
    out=dict(selected_step=r['selected_step'],offline=r['offline'],status=status,closed_loop={},by_start={})
    for mode in ['normal','frozen','clock']:
        out['closed_loop'][mode]={}
        for greedy in [True,False]:
            rows=[e for e in r['episodes'] if e['mode']==mode and (e['sample_seed'] is None)==greedy]
            out['closed_loop'][mode]['greedy' if greedy else 'sampled']=dict(passed=sum(e['target_reached'] for e in rows),episodes=len(rows))
    for ident in sorted({e['start_id'] for e in r['episodes']}):
        out['by_start'][ident]={mode:dict(greedy=next(e['target_reached'] for e in r['episodes'] if e['start_id']==ident and e['mode']==mode and e['sample_seed'] is None),sampled=sum(e['target_reached'] for e in r['episodes'] if e['start_id']==ident and e['mode']==mode and e['sample_seed'] is not None)) for mode in ['normal','frozen','clock']}
    return out

def main():
    status=json.loads((OUTPUT/'status.json').read_text());assert status['state']=='completed'
    selection=status['selection'];summary=dict(selection=selection,checks=json.loads((OUTPUT/'checks.json').read_text()),runs={})
    specs=[(2,0),(4,0)]+([(selection['selected_steps'],s) for s in [1,2]] if selection['replication_gate_passed'] else [])
    for k,seed in specs:
        p=OUTPUT/f'k{k}'/f'graph_seed{seed}';r=result(p);matched=json.loads((p/'matched_checks.json').read_text());assert matched['initial_tensors_identical'] and matched['data_presentations_matched']
        summary['runs'][f'k{k}_seed{seed}']=r
    for seed in sorted({s for k,s in specs}):summary['runs'][f'k1_seed{seed}']=result(ORIGINAL/f'graph_seed{seed}')
    summary['aggregates']={}
    for k in (1,2,4):
        rows=[r for key,r in summary['runs'].items() if key.startswith(f'k{k}_')]
        if not rows:continue
        agg=dict(training_seeds=len(rows),test_balanced_accuracy=sum(r['offline']['test']['normal']['balanced_accuracy'] for r in rows)/len(rows),frozen_balanced_accuracy=sum(r['offline']['test']['frozen']['balanced_accuracy'] for r in rows)/len(rows),closed_loop={})
        for mode in ['normal','frozen','clock']:
            agg['closed_loop'][mode]={}
            for sampling in ['greedy','sampled']:
                agg['closed_loop'][mode][sampling]={field:sum(r['closed_loop'][mode][sampling][field] for r in rows) for field in ['passed','episodes']}
        summary['aggregates'][str(k)]=agg
    summary['new_training_runs']=len(specs);summary['new_evaluation_episodes']=204*len(specs)
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:dict(test_ba=r['offline']['test']['normal']['balanced_accuracy'],frozen_ba=r['offline']['test']['frozen']['balanced_accuracy'],closed_loop=r['closed_loop']) for k,r in summary['runs'].items()},indent=2))
    lines=['## 完成结果','',f"本轮完成{len(specs)}组新训练、{204*len(specs)}局闭环评估，另有每组1局批量/单环境一致性检查。1次更新基线复用原结果。",'','| 内部次数 / seed | 选中更新 | 训练平衡准确率 | 测试正常/固定平衡准确率 | greedy正常/固定/时间表（各17局） | sampled正常/固定/时间表（各51局） |','|---|---:|---:|---:|---|---|']
    for key in sorted(summary['runs']):
        r=summary['runs'][key];o=r['offline'];g=' / '.join(str(r['closed_loop'][m]['greedy']['passed']) for m in ['normal','frozen','clock']);a=' / '.join(str(r['closed_loop'][m]['sampled']['passed']) for m in ['normal','frozen','clock'])
        lines.append(f"| {key} | {r['selected_step']} | {o['train']['normal']['balanced_accuracy']:.1%} | {o['test']['normal']['balanced_accuracy']:.1%} / {o['test']['frozen']['balanced_accuracy']:.1%} | {g} | {a} |")
    lines+=['','预先约定的复验门槛：所选候选的验证集正常视觉平衡准确率至少比旧seed0最佳高5个百分点，且至少比固定首帧高5个百分点。候选按正常视觉验证指标最高者选择，平分取较少内部次数。测试结果不参与选择。','',f"复验门槛结果：{'通过' if selection['replication_gate_passed'] else '未通过'}；所选候选k={selection['selected_steps']}。",'','| 候选 | 验证正常平衡准确率 | 验证固定首帧平衡准确率 |','|---|---:|---:|']
    for k,r in selection['pilots'].items():lines.append(f"| {k} | {r['normal']['balanced_accuracy']:.1%} | {r['frozen']['balanced_accuracy']:.1%} |")
    lines+=['','训练阶段记录的墙钟与显存（含初始和阶段验证，不含模型初始化、数据加载与闭环评估；同预算不等于同算力）：','', '| 内部次数 / seed | 训练秒数 | 峰值allocated MiB | 有效样本呈现量 |','|---|---:|---:|---:|']
    for key in sorted(summary['runs']):
        s=summary['runs'][key]['status'];lines.append(f"| {key} | {s['wall_seconds']:.1f} | {s['peak_allocated_mib']:.1f} | {s['valid_presentations']} |")
    lines.append('')
    report=OUTPUT.parents[1]/'microsteps_results.md';text=report.read_text();marker='## 实验设计'
    if '## 完成结果' in text:text=text[:text.index('## 完成结果')]+text[text.index(marker):]
    report.write_text(text.replace(marker,'\n'.join(lines)+'\n'+marker,1))
if __name__=='__main__':main()
