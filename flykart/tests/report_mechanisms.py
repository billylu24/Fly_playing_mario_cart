"""Generate the architecture hypothesis audit report from recorded artifacts."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/architecture_mechanisms_v1'
def read(name):return json.loads((OUT/name).read_text())
def main():
    assert read('status.json')['state']=='completed'
    meta=read('metadata_audit.json');wash=read('washout.json');val=read('pilot_validation.json');test=read('new_test_offline.json');select=read('pilot_selection.json')
    lines=['# 架构三项假设：数据审计、衰减测量与受控训练试点','','## 设计','','使用同一 k2 seed0 既有检查点；冻结 CNN 和 critic，只训练动作头、节点增益，以及空间组新增的 2048 个投影参数。七组各 128 次监督更新，训练轨迹、抽样次序、类别权重、优化器相同；在 64/128 更新按验证集正常平衡准确率选择（交叉熵打破平局）。这是对既有训练模型的短预算干预，不是从头训练或多种子结论。','','新增 18 个起点：prefix465/525/585 × perturb20/40 × center/left/right；只保留 oracle 达标且 jumps=0 的起点，筛选不参考模型结果。预先规定：候选验证 BA 比基线至少高 5pp，且比自身固定画面高 5pp；空间/符号还要比各自打乱对照高 3pp，才进入闭环。基线和预先指定的旧 CNN 直连正对照始终评估。未过门槛的候选没有闭环测试，不能将它们的离线结果说成驾驶结论。','','## 数据支持范围','',f'当前 {meta["visual_nodes"]:,} 个视觉节点中，{meta["hex_nodes"]:,} 个有 assignedOlHex1/2，覆盖 {meta["hex_fraction"]*100:.1f}%。这些是视叶柱坐标，尚无到游戏屏幕的视角、方向校准。实验按左右侧分别归一化两个坐标，双线性采样冻结 CNN 的 9×9 卷积图；空间打乱在 type×somaSide 内置换完整坐标对。不能把该映射称为真实视网膜输入。','','空间组在原有注入上增加零初始化的 32→64 点投影，只注入有坐标节点；因此初始行为与基线完全一致。空间/打乱两组相同参数量；它们共同增加了卷积特征访问权限，所以空间组相对基线的改善本身不能证明空间组织有效，必须看打乱对照。','',f'按神经元高置信度（≥0.8）的 GABA 预测且与 consensus 一致，标记 {meta["high_confidence_consensus_gaba_nodes"]:,} 个负号发送节点，涉及 {meta["inhibitory_outgoing_edges"]:,} 条出边。其余节点维持基线正号；没有把谷氨酸、组胺、调质和未知类别强行完整解释为 E/I。这是保守 GABA 符号假设。符号打乱在 superclass×somaSide 内进行，保持每组负号节点数；保留所有连接及权重绝对值，因而绝对入边和完全不变，但边加权的负号比例不保证相同。','','## 实际状态衰减','','三对验证轨迹各输入 16 个不同历史画面，再输入完全相同的后续画面，测量全状态差异最大绝对值。下表为相对分叉结束时差异的中位百分比；未使用 Context 每 32 步的人工历史刷新，以隔离网络本身。共同后续输入沿第一条轨迹推进，轨迹结束后保持其最后一帧。','','| 变体 | 4决策后 | 8决策后 | 16决策后 | 32决策后 |','|---|---:|---:|---:|---:|']
    decay={}
    for kind in ['k1','k2','k4','slower','gain1.2']:
        rs=[r for r in wash['results'] if r['variant']==kind];ratios=[float(np.median([r['measurements'][at]['global_linf']/r['measurements'][0]['global_linf'] for r in rs])) for at in [4,8,16,32]];decay[kind]=ratios;lines.append('| '+kind+' | '+' | '.join(f'{x*100:.4f}%' for x in ratios)+' |')
    lines+=['','slower 将更新系数从 0.5 降到 0.2；gain1.2 只将递归增益乘 1.2。两者保持 k2。它们确实延缓上述差异衰减，但不等于当前图像信息每步按此比例丢失，也不证明该任务需要更长记忆。衰减测量未重新训练变体；驾驶学习效果由下述独立训练试点判断。','','## 监督训练结果','','数值为动作平衡准确率（%）。新测试在确定候选后统一解封，各候选都报告，无后续调参。','','| 变体 | 验证正常 | 验证固定 | 新测试正常 | 新测试固定 |','|---|---:|---:|---:|---:|']
    for kind in val:
        vs=[val[kind][m]['balanced_accuracy'] for m in ['normal','frozen']]+[test[kind][m]['balanced_accuracy'] for m in ['normal','frozen']];lines.append('| '+kind+' | '+' | '.join(f'{v*100:.1f}' for v in vs)+' |')
    lines+=['','预定验证门槛结果：'+json.dumps(select['gates'],ensure_ascii=False)+'。','','## 新起点闭环','','| 变体 | 贪心正常 | 贪心固定 | 贪心时间 | 采样正常 | 采样固定 | 采样时间 |','|---|---:|---:|---:|---:|---:|---:|']
    closed={}
    for kind in select['selected']+['cnn']:
        r=read(kind+'_new_evaluation.json');closed[kind]={};cells=[]
        assert r['batched_sequential_match']
        for greedy in [True,False]:
            for mode in ['normal','frozen','clock']:
                es=[e for e in r['episodes'] if (e['sample_seed'] is None)==greedy and e['mode']==mode];n=sum(e['target_reached'] for e in es);key=('greedy_' if greedy else 'sampled_')+mode;closed[kind][key]=dict(success=n,total=len(es));cells.append(f'{n}/{len(es)}')
        lines.append('| '+kind+' | '+' | '.join(cells)+' |')
    old=json.loads((ROOT/'runs/supervised_architecture_v1/manifest.json').read_text());oldhash=set()
    for r in old:
        if r['accepted']:
            with np.load(ROOT/'runs/supervised_architecture_v1/data'/r['filename']) as d:oldhash.update(hashlib.sha256(o.tobytes()).hexdigest() for o in d['observations'])
    new=read('new_test_manifest.json');newhash=set()
    for r in new:
        if r['accepted']:
            with np.load(OUT/'new_test'/r['filename']) as d:newhash.update(hashlib.sha256(o.tobytes()).hexdigest() for o in d['observations'])
    overlap=len(oldhash&newhash)
    lines+=['','## 验证与边界','',f'新起点 oracle 可行 {sum(r["accepted"] for r in new)}/{len(new)}；新旧成功示范的完全相同图像哈希交集为 {overlap}。起点配置不同不代表视觉状态独立，轨迹仍可能在同赛道收敛。','', '空间残差初始前向与原 k2 一致，投影梯度非零；各组 CNN 梯度为空，训练前后 CNN 张量完全不变；固定 W 检查和样本呈现次数匹配通过。通过闭环评估的每个模型均做逐动作批量/顺序一致性检查。原 k2 检查点完整性通过。所有候选均有协议、初始/最佳/末次权重、指标和节点映射文件。','','本轮仅 seed0、128 更新。缺少提升可排除“该具体小预算干预已解决问题”，不能排除更充分训练、完整 E/I 机制、校准视网膜映射或其他动力学。']
    lines+=['','## 结论','','旧 CNN 直连动作头在新增起点仍有强视觉优势：正常 greedy 15/16、固定1/16、时间6/16；采样正常38/48、固定2/48、时间10/48。模型未重新训练。这进一步支持当前短任务的视觉输入可用，失败集中在当前图路径与训练组合，不能归因于新起点整体不可学。','','1. 空间有序与打乱组验证 BA 都为 44.4%，新测试均 44.6%，没有显示空间对应优势。额外检查确认交换坐标确实改变注入（RMS约0.00348）和 logits（RMS约0.000247），并非代码没有施加干预；但覆盖部分节点的弱附加路径不能代表所有空间映射方案。','2. 保守 GABA 符号验证 BA 41.5%，打乱42.1%，均低于基线。新测试 GABA 44.0%、打乱41.3%，单个小预算试点的微小差异不足以确立生物符号收益。符号前向/反向与显式带符号稀疏矩阵一致；原边权绝对值不变。继承全正网络检查点、冻结已适应该网络的 CNN，也限制了这一对照的解释。','3. 实际状态差异会快速衰减；慢更新和增强递归确实延缓衰减。但是慢更新新测试 BA 35.1%；增强递归虽通过验证门槛，新测试 BA 42.7%，并且正常贪心1/16、固定1/16，采样正常8/48、固定13/48，未改善视觉驾驶。记得更久不是当前问题的充分解法。','','下一步不宜据此大规模扫递归增益或直接返回 PPO。若继续研究连接组，优先做固定 CNN 任务表示的传输/重建实验：把已能控制的视觉表示作为辅助监督，检验可训练输入接口能否让 motor 状态重建它，先区分表示传输是否可训练，再研究动作与 RL。该实验尚未执行。']
    (OUT/'summary.json').write_text(json.dumps(dict(decay=decay,closed_loop=closed,new_old_exact_frame_overlap=overlap,selection=select),indent=2)+'\n');(ROOT/'architecture_mechanisms_results.md').write_text('\n'.join(lines)+'\n');print(json.dumps(closed,indent=2))
if __name__=='__main__':main()
