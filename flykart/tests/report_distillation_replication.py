"""Common-start optimizer-order replication with paired controls."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/distillation_replication_v1'
def read(n):return json.loads((OUT/n).read_text())
def main():
    status=read('status.json');assert status['state']=='completed';n=status['new_test_accepted'];kinds=['reconstruction','joint'];summaries={};clock_ref=None;lines=['# 视觉图控制的训练采样种子复验','','## 复验范围','','固定上一轮的重建与联合蒸馏方案，没有修改损失、学习率、模型结构或训练预算。复用seed0检查点，新增seed1/2，每种目标各256次更新；六个图策略在完全相同的新起点上测试。CNN教师及warm-start预训练图相同，只改变minibatch采样序列（样本抽取及顺序，RNG12000+seed）。这检验优化顺序敏感性，不能称为三个独立骨干或三个独立预训练。','','两种目标在每个seed下初值、采样顺序相同；各自按照验证正常教师KL选择0/64/128/192/256检查点，平局取较早。所有模型选择在新测试前完成，不按测试挑模型。','','新增24个候选起点：prefix455/505/565/625 × perturb14/38 × center/left/right，按oracle成功且无跳点保留。较早或较晚的起点可能不被oracle接受；因此泛化结论仅针对保留子集。时间对照和直接CNN正对照所有模型共用。','','预定稳定成功规则：同一目标的三个seed都必须满足greedy正常≥60%、正常比固定高≥20pp、比时间高≥10pp，且每个seed的采样正常通过率都高于固定和时间。规则用于避免只报告最好的一个种子；不满足也完整报告各项结果。','','## 每个种子的闭环结果','','通过指当前弯道检查点，非整场比赛。采样为8100/8101/8102三个种子；同一训练seed各视觉模式的采样种子匹配。','','| 目标 | 训练seed | 贪心正常 | 贪心固定 | 贪心时间 | 采样正常 | 采样固定 | 采样时间 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    gates={};offline=[];pairing={}
    for kind in kinds+['cnn']:
        summaries[kind]={};kind_gates=[]
        for seed in ([0] if kind=='cnn' else [0,1,2]):
            name='cnn' if kind=='cnn' else f'{kind}_seed{seed}';r=read(name+'_evaluation.json');es=r['episodes'];assert len(es)==12*n and r['batched_sequential_match'];clocks=[e['actions'] for e in es if e['mode']=='clock']
            if clock_ref is None:clock_ref=clocks
            else:assert clock_ref==clocks
            metrics={};cells=[]
            for greedy in [True,False]:
                for mode in ['normal','frozen','clock']:
                    sub=[e for e in es if (e['sample_seed'] is None)==greedy and e['mode']==mode];wins=sum(e['target_reached'] for e in sub);key=('greedy_' if greedy else 'sampled_')+mode;metrics[key]=dict(success=wins,total=len(sub));cells.append(f'{wins}/{len(sub)}')
            summaries[kind][str(seed)]=metrics;lines.append('| '+kind+f' | {seed} | '+' | '.join(cells)+' |')
            if kind!='cnn':
                rate=lambda key:metrics[key]['success']/metrics[key]['total'];passed=(rate('greedy_normal')>=.6 and rate('greedy_normal')-rate('greedy_frozen')>=.2-1e-12 and rate('greedy_normal')-rate('greedy_clock')>=.1-1e-12 and rate('sampled_normal')>max(rate('sampled_frozen'),rate('sampled_clock')));kind_gates.append(passed)
                o=r['offline'];offline.append((kind,seed,o));paired={}
                for mode in ['frozen','clock']:
                    a={e['start_id']:e['target_reached'] for e in es if e['sample_seed'] is None and e['mode']=='normal'};b={e['start_id']:e['target_reached'] for e in es if e['sample_seed'] is None and e['mode']==mode};assert a.keys()==b.keys();paired[mode]=dict(normal_only=sum(a[k] and not b[k] for k in a),control_only=sum(b[k] and not a[k] for k in a),both=sum(a[k] and b[k] for k in a),neither=sum(not a[k] and not b[k] for k in a))
                pairing[name]=paired
        if kind!='cnn':gates[kind]=dict(per_seed=kind_gates,all_pass=all(kind_gates))
    lines+=['','## 跨种子汇总','','同一批起点、相同教师被重复使用，下表仅描述总次数，不能视为独立样本扩大了三倍。','','| 目标 | 贪心正常 | 贪心固定 | 贪心时间 | 采样正常 | 采样固定 | 采样时间 | 全部seed过门槛 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for kind in kinds:
        cells=[]
        for key in ['greedy_normal','greedy_frozen','greedy_clock','sampled_normal','sampled_frozen','sampled_clock']:
            wins=sum(x[key]['success'] for x in summaries[kind].values());total=sum(x[key]['total'] for x in summaries[kind].values());cells.append(f'{wins}/{total}')
        lines.append('| '+kind+' | '+' | '.join(cells)+' | '+str(gates[kind]['all_pass'])+' |')
    lines+=['','## 新测试离线动作指标','','| 目标 | seed | 动作BA正常 | 动作BA固定 | 教师KL正常 | 重建MSE正常 |','|---|---:|---:|---:|---:|---:|']
    for kind,seed,o in offline:lines.append(f'| {kind} | {seed} | {o["normal"]["action_scores"]["balanced_accuracy"]*100:.1f}% | {o["frozen"]["action_scores"]["balanced_accuracy"]*100:.1f}% | {o["normal"]["teacher_kl"]:.4f} | {o["normal"]["normalized_mse"]:.4f} |')
    lines+=['','## 检查与边界','',f'候选起点{n}/24被oracle接受。六个图策略和一个CNN正对照各{n*12}局，共{n*12*7}局，另有7局顺序回归。所有模型通过逐动作批量/顺序一致性；共用时间动作完全一致。旧checkpoint、源码和元数据哈希未改变。每个新训练seed的两种目标具有相同初值和有效训练样本呈现次数；不同seed的有效呈现次数可能因轨迹长度不同而不同，这是原采样协议的一部分。','','详细的greedy成对起点表（正常独有成功、对照独有成功、共同成功、共同失败）见summary.json，可区分总体成功率与成对优势。当前只能讨论同赛道的优化顺序稳定性，不能推断真实果蝇拓扑优于随机图，也不能把正常/固定差距增大而正常性能下降称为改进。']
    lines+=['','## 结论','','联合目标的正常greedy通过数为15/20、13/20、11/20，固定画面2/20、5/20、4/20，时间均3/20；采样正常37/60、26/60、28/60，也都超过各自固定和时间对照。视觉收益在三个采样种子下均出现，但seed2正常成功率55%未达到预设60%，所以联合方案未全部通过稳定性门槛。','','仅重建正常greedy14/20、6/20、10/20，两个种子未达60%；联合方案三个seed的greedy正常都更高，且范围较窄。但样本量仅三个采样序列，采样驾驶的优势较小且并非每个seed都胜过重建，不能宣称已确定普遍优越。','','共同CNN正对照greedy16/20、固定1/20、时间3/20，采样46/60、固定13/60、时间13/60。当前图策略仍未达到直接CNN的整体表现，也未证明生物拓扑本身有独特收益。','','新增示范与此前训练/诊断成功示范无完全相同帧（见overlap_audit.json），但仍是同赛道相邻配置。协议和状态中的order是采样序列的简写：有放回抽样也改变样本构成，不是只排列同一个样本多重集。复验没有重新初始化CNN或整个预训练过程。','','三种联合模型都失败但CNN成功的greedy起点为f455_left_p38、f505_left_p14。下一步优先检查这两处首次动作分歧与随后状态偏移，判断是瞬时动作信息损失还是偏离示范后的纠错不足；暂不增加新的抑制/动力学机制。若修复后在未用于诊断的新起点稳定，再做独立预训练及真实/随机拓扑对照。']
    (OUT/'summary.json').write_text(json.dumps(dict(closed_loop=summaries,stability_gates=gates,paired_greedy=pairing),indent=2)+'\n');(ROOT/'distillation_replication_results.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(gates=gates,closed_loop=summaries),indent=2))
if __name__=='__main__':main()
