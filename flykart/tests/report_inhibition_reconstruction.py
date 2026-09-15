"""Report the full-data matched inhibitory reconstruction experiment."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/inhibition_reconstruction_v1'
def read(n):return json.loads((OUT/n).read_text())
def main():
    status=read('status.json');assert status['state']=='completed';names=['positive','gaba','shuffled'];results={k:read(k+'_results.json') for k in names};test=read('new_test_offline.json');selection=read('selection.json')
    lines=['# 完整轨迹的抑制机制与视觉重建对照','','## 设计与可解释范围','','本轮固定同一已训练CNN，重新初始化节点增益a=0、512参数视觉接口残差=0、2129→64线性重建读出=0。三组只改变发送神经元符号：全正、保守高置信GABA负号、在superclass×side内打乱GABA负号。保留全部原连接拓扑和权重绝对值，神经更新均为k2、leak0.5。它不是完整E/I、受体或脉冲神经元模型，也不是对不同更新方程的全面检验。','','使用全部43条训练轨迹（2184决策）和6条分开的验证轨迹（251决策）。同一固定随机N×8接口基底、相同训练顺序、384更新、batch8、BPTT32、最多64决策burn-in；Adam lr0.0003 eps1e-5、clip1。选择0/64/.../384检查点中验证正常重建MSE最低者，平局取较早。动作头固定为之前CNN直连MLP，不使用动作损失训练图。','','CNN特征标准化只用完整训练集；三组共用正连接初始图在训练集的motor均值/标准差（下限1e-5），避免每组额外改变尺度，但这一共同尺度可能对符号组不完全合适。CNN仍来自先前正连接模型，它的表示也可能带有历史训练偏向；本轮只消除了增益和读出的继承。','','新增起点为prefix475/535/595 × perturb16/32 × center/left/right，共18候选。按oracle达标且jumps=0筛选，不参考模型结果。模型选择先完成，再打开新起点测试。正连接和验证MSE最好的符号组必做闭环；若最佳符号组MSE至少比正连接低10%、且其正常动作BA比固定画面高10pp，则连另一符号组也评估。CNN直连正对照始终评估。不是所有模型都做了闭环，未评估的不能报告驾驶结论。','','## 重建与动作信息','','动作BA为三类召回率平均值；重建MSE按训练集CNN特征尺度计算。','','| 模型 | 选择更新 | 训练MSE | 验证MSE正常/固定 | 新测试MSE正常/固定 | 新测试动作BA正常/固定 |','|---|---:|---:|---:|---:|---:|']
    for k,r in results.items():
        v=r['validation'];t=test[k];lines.append(f'| {k} | {r["selected_step"]} | {r["train"]["normal"]["normalized_mse"]:.4f} | {v["normal"]["normalized_mse"]:.4f} / {v["frozen"]["normalized_mse"]:.4f} | {t["normal"]["normalized_mse"]:.4f} / {t["frozen"]["normalized_mse"]:.4f} | {t["normal"]["action_scores"]["balanced_accuracy"]*100:.1f}% / {t["frozen"]["action_scores"]["balanced_accuracy"]*100:.1f}% |')
    lines+=['','选择记录：'+json.dumps(selection,ensure_ascii=False)+'。','','## 新起点闭环驾驶','','重建特征反标准化后送入同一冻结CNN动作头；策略只能通过motor状态读取视觉信息。CNN正对照绕过图网络。正常、固定首帧（整个历史）和共同纯时间线性探针分别测试贪心及三个采样种子。通过为当前弯道任务达标，非整场比赛完赛。','','| 模型 | 贪心正常 | 贪心固定 | 贪心时间 | 采样正常 | 采样固定 | 采样时间 |','|---|---:|---:|---:|---:|---:|---:|']
    closed={};clocks=[];count=status['new_test_accepted']
    for k in status['selection']+['cnn']:
        r=read(k+'_evaluation.json');assert len(r['episodes'])==count*12;assert r['batched_sequential_match'];clocks.append([e['actions'] for e in r['episodes'] if e['mode']=='clock']);closed[k]={};cells=[]
        for greedy in [True,False]:
            for mode in ['normal','frozen','clock']:
                es=[e for e in r['episodes'] if (e['sample_seed'] is None)==greedy and e['mode']==mode];n=sum(e['target_reached'] for e in es);closed[k][('greedy_' if greedy else 'sampled_')+mode]=dict(success=n,total=len(es));cells.append(f'{n}/{len(es)}')
        lines.append('| '+k+' | '+' | '.join(cells)+' |')
    assert all(c==clocks[0] for c in clocks)
    new=read('new_test_manifest.json');newhash=set();oldhash=set()
    for r in new:
        if r['accepted']:
            with np.load(OUT/'new_test'/r['filename']) as d:newhash.update(hashlib.sha256(o.tobytes()).hexdigest() for o in d['observations'])
    original=ROOT/'runs/supervised_architecture_v1'
    for r in json.loads((original/'manifest.json').read_text()):
        if r['accepted']:
            with np.load(original/'data'/r['filename']) as d:oldhash.update(hashlib.sha256(o.tobytes()).hexdigest() for o in d['observations'])
    overlap=len(newhash&oldhash)
    lines+=['','## 梯度、状态与完整性','','| 模型 | 验证状态饱和比例 | 增益参数变化L2 | 接口参数变化L2 | CNN冻结 |','|---|---:|---:|---:|---|']
    for k,r in results.items():lines.append(f'| {k} | {r["validation"]["normal"]["mean_state_saturation"]:.6f} | {r["parameter_updates"]["b.a"]:.4g} | {r["parameter_updates"]["interface"]:.4g} | {r["cnn_unchanged"]} |')
    lines+=['',f'三个模型训练参数初始哈希和有效样本呈现次数相同。正连接初始动力学与参考k2一致；序列重建、目标padding mask和部署Context一致性检查通过；接口/增益梯度非零、CNN梯度为空。源代码、原checkpoint、符号元数据哈希在结束时不变。已评估模型均通过批量/顺序逐动作一致性检查，纯时间序列完全相同。新旧原始成功示范的相同图像哈希交集为{overlap}。', '', '单种子384更新、同赛道新增起点。符号组没有新增可训练参数；若它没有优势，只能说明此保守符号方案在本轮条件下未显示收益，不能排除完整生物抑制机制。重建误差、动作分类和驾驶成功率可能不一致，应分别报告。']
    lines+=['','## 结论与下一步','','1. 在同一可训练接口、完整轨迹、共同初始化和预算下，保守GABA符号没有改善学习：验证MSE正连接0.162、GABA0.178、打乱0.188；新测试动作BA分别45.1%、38.8%、40.6%。GABA比打乱有一点重建优势，但不优于正连接，不能认定递质抑制结构有效。','2. 状态饱和比例GABA约0.155%，正连接约0.308%，都不高；饱和降低没有转成任务收益，不能把缺少抑制导致饱和作为已证实主因。','3. 正连接与GABA闭环正常/固定greedy均为5/16、4/16；采样正连接19/48、18/48，GABA8/48、8/48，均缺乏可靠的持续视觉收益。CNN正对照greedy13/16、3/16，采样40/48、7/48，说明新起点仍可依赖视觉解决。','4. 表示MSE下降不能替代动作与闭环目标。本轮重建预训练后直接接冻结CNN动作头，没有再用动作监督适配motor读出；不能把结果解释成所有动作适配路线已经失败。','','下一项建议优先比较同一正连接可训练接口下的“仅特征重建”与“教师动作分布蒸馏＋特征重建”，让训练直接约束动作相关差异；保持CNN教师冻结、无CNN到动作的旁路，并在新的起点闭环验证。先不扩大抑制参数搜索，不引入多个新的神经动力学机制。这个建议来自重建与控制指标的分离，尚未验证蒸馏一定有效。']
    (OUT/'summary.json').write_text(json.dumps(dict(closed_loop=closed,exact_frame_overlap_original=overlap,selection=selection),indent=2)+'\n');(ROOT/'inhibition_reconstruction_results.md').write_text('\n'.join(lines)+'\n');print(json.dumps(closed,indent=2))
if __name__=='__main__':main()
