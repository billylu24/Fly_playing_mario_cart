"""Insert completed matched-seed results into the experiment report."""
import json
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'runs/supervised_architecture_v1/summary.json').read_text())
lines=['## 完成结果','', '全部6组监督训练完成，每组256次更新，全部所选checkpoint完成204局闭环评估，共1224局；另有6局单环境与批量评估器一致性检查。图模型W与所选checkpoint完整性检查通过。','', '| 架构 | 测试平衡准确率（3seed均值±样本标准差） | 固定首帧平衡准确率 | 正常视觉greedy通过 | 固定首帧greedy通过 | 时间表greedy通过 |','|---|---:|---:|---:|---:|---:|']
for arch,label in [('graph','固定脑网络'),('gru','CNN+GRU')]:
 a=s['aggregates'][arch];n=a['normal_test_balanced_accuracy'];f=a['frozen_test_balanced_accuracy'];values=[]
 for mode in ['normal','frozen','clock']:
  x=a[mode+'_greedy'];values.append(f"{x['passed']}/{x['episodes']}")
 lines.append(f"| {label} | {n['mean']:.1%} ± {n['sample_std']*100:.1f}个百分点 | {f['mean']:.1%} | "+' | '.join(values)+' |')
lines+=['','| 架构 / seed | 选中更新 | 训练平衡准确率 | 测试正常/固定平衡准确率 | 正常/固定/时间表greedy通过（各17局） |','|---|---:|---:|---:|---|']
for arch in ['graph','gru']:
 for seed in [0,1,2]:
  r=s['runs'][f'{arch}_seed{seed}'];o=r['offline'];counts=[str(r['closed_loop'][m]['greedy']['passed']) for m in ['normal','frozen','clock']]
  lines.append(f"| {arch} / {seed} | {r['selected_step']} | {o['train']['normal']['balanced_accuracy']:.1%} | {o['test']['normal']['balanced_accuracy']:.1%} / {o['test']['frozen']['balanced_accuracy']:.1%} | "+' / '.join(counts)+' |')
lines+=['','随机采样动作另报，不与greedy相加解释：','', '| 架构 | 正常视觉 | 固定首帧 | 纯时间表 |','|---|---:|---:|---:|']
for arch in ['graph','gru']:
 values=[]
 for m in ['normal','frozen','clock']:
  a=s['aggregates'][arch][m+'_sampled'];values.append(f"{a['passed']}/{a['episodes']}")
 lines.append('| '+arch+' | '+' | '.join(values)+' |')
lines+=['','上述分母包含同一批17个测试起点在3个训练seed上的重复评估；采样模式还包含每起点3个动作seed，不代表153个独立起点。没有按episode独立假设报告显著性。平衡准确率把三类等权，避免左转标签占多数造成虚高。','', '各起点greedy通过数汇总（每格为3个训练seed的通过数）：','', '| 起点 | graph正常/固定/时间表 | GRU正常/固定/时间表 |','|---|---|---|']
ids=sorted(s['runs']['graph_seed0']['by_start'])
for ident in ids:
 vals=[]
 for arch in ['graph','gru']:
  vals.append(' / '.join(str(sum(s['runs'][f'{arch}_seed{seed}']['by_start'][ident][mode]['greedy'] for seed in [0,1,2])) for mode in ['normal','frozen','clock']))
 lines.append('| '+ident+' | '+' | '.join(vals)+' |')
lines+=['','结果文件：`runs/supervised_architecture_v1/summary.json`含汇总与逐起点结果，6个子目录分别保留initial/best/last checkpoint、训练曲线和全部闭环动作。','']
p=root/'supervised_architecture_results.md';text=p.read_text();marker='## 预定设计';
if '## 完成结果' in text:text=text[:text.index('## 完成结果')]+text[text.index(marker):]
text=text.replace(marker,'\n'.join(lines)+'\n'+marker,1);p.write_text(text)
