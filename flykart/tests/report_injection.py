"""Render the completed projection ablation into the project report."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'runs/visual_injection_v1/summary.json').read_text())
lines=['## 完成结果','','3个训练seed均完成256更新与204局闭环评估，共612局；另外3局批量/单环境一致性检查通过。共享初始tensor、数据及有效样本呈现量逐seed匹配，图W与所选checkpoint完整性检查通过。','', '| 模型 | 测试平衡准确率（均值±样本标准差） | 固定首帧平衡准确率 | 正常/固定/时间表greedy通过（各51局） |','|---|---:|---:|---|']
for k,label in [('graph','旧固定接口'),('injection','可训练分组接口'),('gru','既有GRU正对照')]:
 a=s['aggregates'][k];x=a['normal_test_balanced_accuracy'];f=a['frozen_test_balanced_accuracy']
 passed=' / '.join(str(a[m+'_greedy']['passed']) for m in ['normal','frozen','clock'])
 lines.append(f"| {label} | {x['mean']:.1%} ± {100*x['sample_std']:.1f}个百分点 | {f['mean']:.1%} | {passed} |")
lines+=['','新接口逐seed结果：','', '| seed | 选中更新 | 训练平衡准确率 | 测试正常/固定平衡准确率 | 正常/固定/时间表greedy通过（各17局） | 投影Delta最大绝对值 |','|---|---:|---:|---:|---|---:|']
for seed,r in s['runs'].items():
 o=r['offline'];passed=' / '.join(str(r['closed_loop'][m]['greedy']['passed']) for m in ['normal','frozen','clock'])
 lines.append(f"| {seed} | {r['selected_step']} | {o['train']['normal']['balanced_accuracy']:.1%} | {o['test']['normal']['balanced_accuracy']:.1%} / {o['test']['frozen']['balanced_accuracy']:.1%} | {passed} | {r['injection']['projection_delta_max']:.5f} |")
lines+=['','随机采样动作（每起点3个动作seed）另报：','', '| 模型 | 正常视觉 | 固定首帧 | 纯时间表 |','|---|---:|---:|---:|']
for k in ['graph','injection','gru']:
 a=s['aggregates'][k];vals=[f"{a[m+'_sampled']['passed']}/{a[m+'_sampled']['episodes']}" for m in ['normal','frozen','clock']]
 lines.append('| '+k+' | '+' | '.join(vals)+' |')
lines+=['','同样17个测试起点在三个训练seed上重复评估，51/153不代表独立起点数。没有据此声称统计显著性或跨赛道泛化。','', '新接口逐起点greedy结果（各3个训练seed）：','', '| 起点 | 正常 | 固定首帧 | 纯时间表 |','|---|---:|---:|---:|']
for ident in sorted(s['runs']['0']['by_start']):
 vals=[str(sum(r['by_start'][ident][mode]['greedy'] for r in s['runs'].values())) for mode in ['normal','frozen','clock']]
 lines.append('| '+ident+' | '+' | '.join(vals)+' |')
lines.append('')
p=root/'visual_injection_results.md';text=p.read_text();marker='## 接口设计'
if '## 完成结果' in text:text=text[:text.index('## 完成结果')]+text[text.index(marker):]
p.write_text(text.replace(marker,'\n'.join(lines)+'\n'+marker,1))
