"""Report reconstruction fit, gradients and limitations."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/transport_reconstruction_v1'
def read(n):return json.loads((OUT/n).read_text())
def main():
    status=read('status.json');assert status['state']=='completed';refs=read('references.json');opt=read('optimizer_controls.json');names=['decoder_only','gain','interface','interface_edges'];results={k:read(k+'.json') for k in names}
    lines=['# 固定视觉表示的图网络传输与梯度诊断','','## 实验问题与设计','','这次不训练驾驶动作，而要求 motor 隐状态重建同一时刻冻结 CNN 的 64 维特征。目标是区分小样本拟合、优化条件、可训练通路和泛化；重建成功本身不代表驾驶成功。','','使用既有 k2 seed0 检查点，固定 CNN。训练为三个预定起点各前24帧，共72目标；验证为六条不同轨迹各前21帧，共126目标（部分验证轨迹不足24帧）。每个序列从零状态开始，全展开24决策/48神经更新，没有分块截断或Context刷新。目标按训练集均值/标准差标准化；motor也按初始训练状态固定标准化。所有组使用同一个零初始化2129→64线性读出，400次全批量Adam，lr0.003、eps1e-5、全局clip1。只报告最终模型，不按验证集挑选。','','四组依次为：只训练读出；读出＋节点增益；再增加可训练视觉接口；再增加受限边权调整。视觉接口使用固定随机N×8基底与可训练8×64投影，残差从零开始，只注入视觉节点；它使用当前冻结CNN特征作为输入，不使用未来帧或标签。受限连接选择现有visual→motor边中每个接收节点权重最大的至多8条，共6092条，各边可在原权重±50%内变化，无新边。这不是全连接组边权训练。','','## 最终重建结果','','标准化MSE越低越好；0预测对应训练均值。验证R²按验证目标自身均值计算，不能简单把1−MSE当作R²。固定画面对照使用该序列首帧特征反复输入，但仍重建正常画面的目标。','','| 训练参数 | 参数量 | 训练MSE | 验证MSE | 验证R² | 固定画面验证MSE |','|---|---:|---:|---:|---:|---:|']
    summary={}
    for name,r in results.items():
        last=r['history'][-1];summary[name]=dict(train=last['train'],validation=last['validation'],frozen=r['interventions']['validation_frozen']);lines.append(f'| {name} | {r["trainable_parameters"]} | {last["train"]["normalized_mse"]:.4f} | {last["validation"]["normalized_mse"]:.4f} | {last["validation"]["r2_global"]:.3f} | {r["interventions"]["validation_frozen"]["normalized_mse"]:.4f} |')
    lines+=['',f'训练均值常数基线：训练MSE {refs["constant_train_mse"]:.4f}，验证MSE {refs["constant_validation_mse"]:.4f}。预定小集拟合门槛为训练MSE≤0.05，各组结果：'+json.dumps(status['training_fit_gate'])+'。','','## 固定motor状态的闭式线性参考','','同一motor状态无需更新图网络即可进行ridge线性拟合；全部候选如下，不隐藏不利结果。最小训练误差用于选择训练插值参考，并非按验证集挑最优模型。','','| ridge | 训练MSE | 验证MSE |','|---|---:|---:|']
    for k,r in refs['ridge'].items():lines.append(f'| {k} | {r["train_mse"]:.8g} | {r["validation_mse"]:.4f} |')
    lines+=['','注意：2129维特征拟合72个样本是高度欠定问题，近乎零训练误差不证明恢复了通用视觉语义，也不能证明连接组具有独特价值；它说明在这些固定样本上，线性读出函数类的表达能力不是无法拟合的充分解释。弱正则插值的验证误差很高，显示过拟合。','','## 优化步长对照','','主实验第一步把误差从0.986提高到34.90，因而追加固定特征、相同零初值和预算的步长对照；预先记录三个学习率，全部报告。预计算与序列计算的浮点执行路径不同，迭代后结果可能偏离，因此比较学习率时使用下表内部匹配的结果。','','| 学习率 | 400步训练MSE | 验证MSE |','|---|---:|---:|']
    for k,h in opt['results'].items():r=h[-1];lines.append(f'| {k} | {r["train_mse"]:.4f} | {r["validation_mse"]:.4f} |')
    lines+=['','这表明固定预算下读出优化对步长敏感；不能把Adam尚未拟合的误差全部归因于图内部信息消失。','','## 梯度与实际参数变化','','在更新0/1/25/100/200/400时，单独用序列最后一个决策的重建损失做梯度探针。它测量不同时间和节点的敏感性，区别于用于优化的全部24决策平均损失。零初始化读出在第0步对上游梯度必然为零，不是断图错误。','','| 组别 | 末步视觉注入梯度RMS | 最早/末步视觉注入梯度比 | 增益参数更新L2 | 接口参数更新L2 | 边参数更新L2 |','|---|---:|---:|---:|---:|---:|']
    allowed=read('allowed_gradient_audit.json')
    for name,r in results.items():
        a=r['gradient_audits']['400'];g=allowed[name]['visual_signal_gradient_rms'];u=r['parameter_updates'];get=lambda k:f'{u[k]["update_l2"]:.4g}' if k in u else '冻结'
        lines.append(f'| {name} | {g[-1]:.4g} | {g[0]/max(g[-1],1e-30):.4g} | {get("b.a")} | {get("interface")} | {get("delta")} |')
    lines+=['','参数发生变化不等于学到了有用信息，梯度非零也不等于优化有效；需结合重建误差和固定画面对照判断。早期梯度较小可能反映收缩，也可能因为重建当前画面不需要很久以前的输入，不能单凭比值认定有害梯度消失。所有节点分组/时间梯度明细保存在各组JSON。','','## 验证与边界','','CNN权重及梯度隔离、原checkpoint完整性检查通过。四组初始输出和第1步误差相同，额外接口和边变化都从零开始。目标标准化统计只来自训练子集；本轮没有重训CNN、没有PPO、没有新的驾驶成功率。固定小集不是完整数据分布，验证轨迹此前已用于多轮诊断，也不构成新的独立泛化测试。']
    lines+=['','## 重建特征的动作可用性','','将重建的CNN特征反标准化后，交给之前冻结的有效CNN动作头；不更新该动作头。只检查这批短窗口上的离线动作，未做闭环驾驶。','','| 组别 | 训练动作BA | 验证动作BA | 验证与原CNN动作一致率 |','|---|---:|---:|---:|']
    for name,a in allowed.items():
        tr=a['downstream_actions']['train'];vr=a['downstream_actions']['validation'];lines.append(f'| {name} | {tr["reconstructed"]["balanced_accuracy"]*100:.1f}% | {vr["reconstructed"]["balanced_accuracy"]*100:.1f}% | {vr["agreement_with_direct_cnn"]*100:.1f}% |')
    direct=allowed['decoder_only']['downstream_actions'];lines+=['',f'直接CNN动作头在相同短窗口上的训练/验证BA为 {direct["train"]["direct_cnn"]["balanced_accuracy"]*100:.1f}% / {direct["validation"]["direct_cnn"]["balanced_accuracy"]*100:.1f}%，只作同数据参考。','', '最终梯度表只统计真正允许注入的视觉节点，并额外记录对64维输入特征的梯度；原始全节点signal梯度含不实际开放的非视觉坐标，不能当作真实视觉输入梯度。新增接口/边权的方向导数通过有限差分检查。']
    lines+=['','## 本轮结论','','1. 没有发现新增接口或受限边权的反向传播错误：方向导数有限差分相对误差小于0.04%，各组允许训练的参数确有更新。上游梯度在零初始化读出之后出现；跨23个决策的输入梯度较小，但这不单独证明有害梯度消失。','2. 当前Adam训练存在优化条件问题：初步更新过冲，减小学习率改善固定特征拟合；同一固定motor状态的线性参考能插值72个训练点。因此不能把原方案的残余训练误差直接解释为图状态没有信息。','3. 相同400更新预算下，开放512参数的视觉接口使训练MSE从仅增益0.213降到0.023，验证从0.214降到0.125；固定首帧验证误差升到1.414，说明利用了持续视觉变化。受限调整6092条边没有进一步改善本轮最终结果。','4. 接口组重建特征经冻结动作头后，验证短窗口动作BA52.9%，仅增益36.3%、只读出26.6%；但直接CNN76.2%，仍有明显任务信息损失。此处是126个验证决策的离线指标，不是驾驶成功率。','5. 正则ridge参考的某些候选验证重建误差比接口组还低（例如ridge0.1为0.102）；不能宣称新接口是唯一或最优解。该候选未按预定训练插值规则选中，且扫描了多个正则值，需独立验证。','','下一步应把完整训练轨迹上的正则读出基线与接口重建预训练做匹配对照，再用动作监督或冻结动作头接入闭环；在新起点检验正常/固定/时间控制。暂不需要开放全部2556万条边或返回PPO。本轮证据支持优化条件和输入接口是可改进因素，尚未证明网络规模本身是失败原因，也未证明真实连接组优于随机图。']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(ROOT/'transport_reconstruction_results.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
