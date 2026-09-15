"""Summarize the matched frozen motor-readout intervention."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/readout_conditioning_v1'
def read(name):return json.loads((OUT/name).read_text())
def main():
    assert read('status.json')['state']=='completed'
    offline=read('offline.json');summary={}
    lines=['# 图网络动作读出尺度对照','','冻结 k2 seed0 的 CNN、增益和固定图，只训练动作头。完整 motor 线性读出的两组使用相同随机初值、训练数据、类别权重、优化器和 600 次更新预算，仅输入是否按训练集均值/标准差标准化不同。各自在每 25 次更新时按验证集正常视觉平衡准确率选检查点（交叉熵打破平局）。标准化模型直接复用上一轮已训练权重，避免重复训练。另评估上一轮的标准化 motor256/MLP32 作为减小维度并增加非线性的对照；它同时改变容量和输入节点，不能单独归因于其中之一。','','全部动作头仅接收 motor 状态，没有 CNN 直连。冻结首帧作用于整个历史，纯时间沿用上一轮训练且按验证集选定的线性时间探针。三个候选全部闭环评估，没有按本轮测试成绩筛选。仍是单个骨干种子、同赛道且复用起点的诊断。','','## 离线动作解码','','数值为三类召回率平均值（%）。','','| 读出 | 选择更新 | 训练正常 | 验证正常 | 测试正常 | 测试固定 |','|---|---:|---:|---:|---:|---:|']
    for name,r in offline.items():
        o=r['offline'];vals=[o[s]['normal']['balanced_accuracy']*100 for s in ['train','validation','test']]+[o['test']['frozen']['balanced_accuracy']*100]
        lines.append('| '+name+f' | {r["selected_step"]} | '+' | '.join(f'{x:.1f}' for x in vals)+' |')
    lines+=['','## 闭环驾驶','','每个候选为 17 起点 × 3 视觉/时间模式 ×（贪心及 3 采样种子），共 204 局；三个候选合计 612 局，另有 3 局批量/顺序一致性回归。通过指达到当前弯道检查点，非完整比赛完赛。','','| 读出 | 贪心正常 | 贪心固定 | 贪心时间 | 采样正常 | 采样固定 | 采样时间 |','|---|---:|---:|---:|---:|---:|---:|']
    clocks=[]
    for name in offline:
        es=read(name+'_evaluation.json')['episodes'];assert len(es)==204;cells=[];summary[name]={}
        clocks.append([e['actions'] for e in es if e['mode']=='clock'])
        for greedy in [True,False]:
            for mode in ['normal','frozen','clock']:
                subset=[e for e in es if (e['sample_seed'] is None)==greedy and e['mode']==mode];wins=sum(e['target_reached'] for e in subset);key=('greedy_' if greedy else 'sampled_')+mode
                summary[name][key]=dict(success=wins,total=len(subset));cells.append(f'{wins}/{len(subset)}')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    assert clocks[0]==clocks[1]==clocks[2]
    scales=read('feature_scales.json')
    lines+=['','## 完整性与解释范围','',f'motor 特征的训练标准差五分位摘要（min/Q1/median/Q3/max）为 {scales["motor_std_quantiles"]}，{scales["below_floor"]}/{scales["dimension"]} 个节点低于标准化下限 1e-5。','', '两种线性头的初始权重完全相同；归一化参数只使用训练集。复用标准化探针重新提取特征后，各划分两种视觉模式的混淆矩阵均与上一轮一致。三种部署读出均通过特征/输出一致性和首起点逐动作批量/顺序回归。骨干参数、原检查点均未改变，骨干梯度为空，三个候选的纯时间动作序列完全相同。','','标准化线性读出在函数表达能力上可吸收到原线性权重和偏置中；其收益若存在，反映当前有限预算下的优化条件变化，而不是新增视觉信息。此实验冻结骨干，不检验端到端训练时标准化对 CNN/增益梯度的影响。']
    lines+=['','## 结论与下一步','','标准化完整线性读出将测试 BA 从 45.6% 提高到 48.4%，但 greedy 正常驾驶从 5/17 变为 3/17、采样从 8/51 变为 6/51，没有转化为闭环改善。motor256/MLP 正常 greedy 为 9/17，固定画面也有 8/17；正常采样 7/51，反而低于固定画面 20/51，仍不能视为可靠视觉反馈。','','上一轮相同 k2 骨干的 CNN 直连 MLP 正常 greedy 为 15/17、固定 4/17，采样正常 40/51、固定 16/51。保留图路径后的结果明显较弱，说明仅事后修正动作读出仍不够；既有图状态的可控性和行为克隆的分布偏移仍需区分。','','下一项建议做匹配的端到端监督对照：同一 k2 初始骨干与动作头初值，原始 motor 读出对比训练集固定统计标准化读出，允许 CNN 和增益共同更新，保持数据/预算一致。它检验改善读出尺度能否改善传回图网络的学习信号；本轮冻结骨干没有回答此问题。继续保留 CNN 直连正对照、正常/固定/时间闭环，并在新增起点上验证后才决定是否返回 PPO。尚未启动这项端到端实验。']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(ROOT/'readout_conditioning_results.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
