"""Summarize all prespecified policies and paired new-start outcomes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/recovery_training_v1'
KINDS=['distill','old_ce','recovery_ce','warm']


def run():
    status=json.loads((OUT/'status.json').read_text());assert status['state']=='completed'
    n=status['new_test_accepted'];models={};episodes={}
    for name in [f'{k}_seed{s}' for s in range(3) for k in KINDS]+['cnn']:
        report=json.loads((OUT/(name+'_evaluation.json')).read_text());assert report['batched_sequential_match'];es=report['episodes'];assert len(es)==12*n
        assert len({(e['start_id'],e['mode'],e['sample_seed']) for e in es})==12*n
        episodes[name]=es;metrics={}
        for sampling in ['greedy','sampled']:
            for mode in ['normal','frozen','clock']:
                rr=[r for r in es if r['mode']==mode and (r['sample_seed'] is None)==(sampling=='greedy')]
                metrics[f'{sampling}_{mode}']=dict(success=sum(r['target_reached'] for r in rr),total=len(rr),rate=sum(r['target_reached'] for r in rr)/len(rr))
        models[name]=metrics
    paired=[]
    for s in range(3):
        for start in sorted({e['start_id'] for e in episodes[f'warm_seed{s}']}):
            paired.append(dict(seed=s,start=start,**{k:next(e['target_reached'] for e in episodes[f'{k}_seed{s}'] if e['start_id']==start and e['mode']=='normal' and e['sample_seed'] is None) for k in KINDS}))
    total=lambda kind,key:sum(models[f'{kind}_seed{s}'][key]['success'] for s in range(3))
    delta=total('recovery_ce','greedy_normal')-total('old_ce','greedy_normal')
    checks=dict(all_seeds_no_worse_than_old_ce=all(models[f'recovery_ce_seed{s}']['greedy_normal']['success']>=models[f'old_ce_seed{s}']['greedy_normal']['success'] for s in range(3)),
        aggregate_gain_at_least_10pp=delta/(3*n)>=.10,
        all_seeds_vision_gain=all(models[f'recovery_ce_seed{s}']['greedy_normal']['rate']-models[f'recovery_ce_seed{s}']['greedy_frozen']['rate']>=.20-1e-12 and models[f'recovery_ce_seed{s}']['greedy_normal']['rate']-models[f'recovery_ce_seed{s}']['greedy_clock']['rate']>=.10-1e-12 for s in range(3)),
        aggregate_sampled_gain=total('recovery_ce','sampled_normal')>total('old_ce','sampled_normal'))
    summary=dict(models=models,paired_greedy=paired,gate_checks=checks,gate_passed=all(checks.values()),aggregate_recovery_minus_old_ce_successes=delta)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 恢复数据训练：损失匹配对照', '',f"完成 9 个模型训练、{status['episodes']} 局闭环评估和 13 局顺序回归校验；新测试接受 {n}/24 个预设候选。", '',
    '三个条件各追加 128 次更新，batch8、BPTT32、burn64、Adam lr0.0003。每个 seed 内从相同已归档 joint 模型开始；三个 seed 共享 CNN/教师和最初预训练，只在此前训练采样顺序上不同。CNN、动作头、拓扑和动力学冻结设置不变，仅增益、视觉接口和重建解码器可训练。', '',
    '- A / distill：两半批次都使用旧数据 MSE+教师 KL。',
    '- B / old_ce：一半保持 MSE+KL，另一半旧数据 MSE+示范动作 CE。',
    '- C / recovery_ce：与 B 相同损失和比例，但动作 CE 半批次换为恢复示范。', '',
    '恢复半批次的前缀/尾部监督掩码同步用于 A/B 对应半批次，确保每个更新的有效监督数量完全匹配。A/B 辅助部分从有完整32帧的旧块采样；因此 A 是本轮匹配对照，不是原训练采样机制的逐项复刻。恢复示范来自三个旧种子训练起点的合并池。', '',
    '所有模型按原验证集正常画面的 oracle 动作交叉熵选 checkpoint（0/32/64/96/128，平局较早），不按新测试表现挑选。', '',
    '## 闭环结果', '',
    '| 模型 | 贪心正常 | 贪心固定图像 | 贪心时间 | 采样正常 | 采样固定图像 | 采样时间 |',
    '|---|---:|---:|---:|---:|---:|---:|']
    keys=['greedy_normal','greedy_frozen','greedy_clock','sampled_normal','sampled_frozen','sampled_clock']
    for name,m in models.items():lines.append('| '+name+' | '+' | '.join(f"{m[k]['success']}/{m[k]['total']}" for k in keys)+' |')
    lines+=['', '## 预设判断', '',f"恢复数据收益门槛：{'通过' if summary['gate_passed'] else '未通过'}。C 相比 B 的贪心正常成功数差为 {delta:+d}/{3*n}（{delta/(3*n):+.1%}）。", '']
    for k,v in checks.items():lines.append(f'- {k}: {v}')
    lines+=['', '门槛要求三个 seed 的 C 不低于 B、合计至少提升10个百分点、每个 C 正常比固定图像高20个百分点且比时间高10个百分点、采样正常合计也高于 B。以上是描述性门槛，不是独立样本显著性检验。所有模型共享测试起点，不把合计局数当作独立环境样本。', '',
    '新起点仍在同一赛道，邻近历史起点；不能宣称跨赛道泛化或图拓扑优于其他架构。恢复标签是特权控制器成功轨迹的动作，可能有视觉不可辨别或非最优动作；本轮只能检验这套固定恢复训练方案。', '',
    '## 验证与复现', '',
    '训练前检查有效监督数量匹配、恢复前缀掩码、BPTT与部署Context输出一致，以及接口/增益/解码器非零有限梯度。CNN和动作头参数保持不变。全部13种策略的第一起点采样正常动作、终止原因、帧数、进度和跳跃数，与顺序回放逐项一致。来源哈希校验通过。', '',
    '代码：`training/recovery_training.py`。预设方案、采样计划、验证记录、完整动作轨迹和模型选择记录位于 `runs/recovery_training_v1/`。ROM、画面NPZ和checkpoint留在本地。', '']
    selected=json.loads((OUT/'selection.json').read_text())['models']
    lines += ['## 训练是否改变了选中模型', '', '| seed | A 选中步数 | B 选中步数 | C 选中步数 |', '|---:|---:|---:|---:|']
    for seed in range(3):lines.append(f"| {seed} | {selected[f'distill_seed{seed}']} | {selected[f'old_ce_seed{seed}']} | {selected[f'recovery_ce_seed{seed}']} |")
    lines += ['', '步数0表示保留起始模型，不能解释为追加训练产生了相同质量的新模型。此轮结果同时取决于训练方案与预设选模规则；没有测试所有末步模型，因此不能推断任意 checkpoint 都无效。', '']
    audit=json.loads((OUT/'teacher_target_audit.json').read_text())['groups']
    lines += ['## 补充诊断与下一步', '',
        f"训练后只读检查：冻结直接 CNN 与旧训练示范动作一致率 {audit['original_train']['teacher_agreement']:.1%}，与去重恢复示范一致率 {audit['recovery_train']['teacher_agreement']:.1%}。在精确重建 CNN 特征时，冻结动作头仍会与部分恢复标签不同，因此重建和动作监督存在目标取舍。这不证明训练不可行，解码器仍可偏离原特征以改变动作。", '',
        'seed1 的恢复组与 B 贪心成功数相同，但救回 f515_left_p26、f515_right_p26，同时丢掉 f445_center_p26、f445_left_p26。采样正常从29/57到43/57，固定图像也从9/57到18/57；这是单种子的行为变化信号，不能作为稳定视觉提升的结论。', '',
        '下一步优先做小规模动作头适应诊断：冻结 CNN、图状态更新及重建解码器，只训练动作头；使用同样训练/恢复数据，同时给直接 CNN 配置相同可训练动作头作对照。另保留当前冻结动作头基线。这样先判断恢复标签是否能从现有视觉/图读出特征学到，再决定是否值得端到端改变目标。', '',
        '验证恢复动作泛化需要另从原验证起点生成恢复验证集，保持训练/验证前缀隔离，并同时监控旧验证行为。当前19个测试起点只保留为诊断集，下一轮调参不得以其成败选模型；需预留新的最终评估起点。若动作头可学习而端到端不稳定，再以匹配对照测试恢复样本上的MSE权重或更小步长；尚不据此增加E/I机制。', '',
        '新测试707个不同画面与旧训练、恢复训练、原验证/测试的精确重叠均为0；仍是同赛道。完整校验见 overlap_audit.json 和 teacher_target_audit.json。', '']
    (ROOT/'recovery_training_results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=status,checks=checks,delta=delta,models=models),indent=2))

if __name__=='__main__':run()
