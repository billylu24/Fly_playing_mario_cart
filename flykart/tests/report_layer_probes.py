"""Build an auditable summary of frozen layer probes."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/layer_probes_v1'
def read(p):return json.loads(p.read_text())
def main():
    results={k:read(OUT/f'{k}_results.json') for k in ['k1','k2','gru']}
    lines=['# 冻结模型的分层动作解码诊断','','本实验冻结已有 seed0 检查点，只训练动作探针。数据沿用原行为克隆轨迹划分：训练 2184、验证 251、测试 795 个决策；测试起点此前已用于诊断，因此结果不能当作新的独立泛化证据。','','## 方法','','CNN 特征为 64 维；图网络分别抽取 256 个视觉注入节点、256 个非视觉且非 motor 节点、256 个 motor 节点，并额外检查完整 2129 维 motor。节点按预定随机种子抽取，没有使用动作标签；这些集合不是严格的解剖学层级。GRU 另检查其 64 维隐藏状态。','','每组训练线性和单隐藏层 32 单元 Tanh 探针，输入按训练集正常视觉的均值/标准差标准化（标准差下限 1e-5）。600 次全批量 Adam 更新，每 25 次按验证集正常视觉平衡准确率选检查点，交叉熵打破平局。类别权重仅用训练标签。固定画面将整个历史替换为该轨迹第一帧；时间基线仅接收决策序号，不接收图像或起点 ID。所有探针只在正常轨迹上训练。','','探针容量和训练预算与原策略不同，不能把提升全部归因于标准化；完整 motor 的输入维数也远大于 CNN。平衡准确率是三类动作召回率的平均值，恒定动作基线为 33.3%。','','## 离线结果','','下表为百分比；每行都是独立训练的探针，未按测试成绩筛选。','','| 骨干 | 特征/探针 | 参数 | 训练正常 | 验证正常 | 测试正常 | 测试固定 |','|---|---|---:|---:|---:|---:|---:|']
    for kind,r in results.items():
        for name,v in r.items():
            if name=='checks':continue
            vals=[v[s]['normal']['balanced_accuracy']*100 for s in ['train','validation','test']]+[v['test']['frozen']['balanced_accuracy']*100]
            lines.append(f'| {kind} | {name} | {v["parameters"]} | '+' | '.join(f'{x:.1f}' for x in vals)+' |')
    t=read(OUT/'time_results.json');chosen=t['selected'];lines+=['',f'纯时间探针按验证集选择 {chosen}：测试平衡准确率 {t["results"][chosen]["test"]["balanced_accuracy"]*100:.1f}%。']
    manifest=read(ROOT/'runs/supervised_architecture_v1/manifest.json');counts={}
    for split in ['train','validation','test']:
        labels=[]
        for r in manifest:
            if r['split']==split and r['accepted']:
                with np.load(ROOT/'runs/supervised_architecture_v1/data'/r['filename']) as d:labels.extend(d['actions'].tolist())
        counts[split]=np.bincount(labels,minlength=3)
    majority=int(counts['train'].argmax());constant={s:dict(action=majority,accuracy=float(c[majority]/c.sum()),balanced_accuracy=1/3) for s,c in counts.items()}
    (OUT/'constant_results.json').write_text(json.dumps(constant,indent=2)+'\n')
    lines+=['',f'训练多数类恒定动作 {majority}：测试普通准确率 {constant["test"]["accuracy"]*100:.1f}%，但平衡准确率只有 33.3%，说明普通准确率容易受类别不平衡误导。','','## 闭环结果','','按预定规则，只使用验证集选择最佳图网络探针和最佳 GRU 探针，并要求正常视觉 BA ≥45%、比固定画面高至少 5 个百分点。选中 k2/CNN/MLP32 和 GRU/隐藏状态/MLP32。k2 探针直接读取 CNN，因此其成绩表示绕过图网络后的控制能力，不是图网络已经学会驾驶。','','每个模型测试 17 个起点 ×（贪心和 3 个采样种子）×（正常、固定、纯时间），共 204 局。纯时间使用上述验证集选出的时间探针，因此和旧报告的“固定训练图像生成动作时间表”并非同一个基线。','','| 策略 | 贪心正常 | 贪心固定 | 贪心时间 | 采样正常 | 采样固定 | 采样时间 |','|---|---:|---:|---:|---:|---:|---:|']
    summary={}
    for kind in ['k2','gru']:
        report=read(OUT/f'{kind}_evaluation.json');summary[kind]={};cells=[]
        for greedy in [True,False]:
            for mode in ['normal','frozen','clock']:
                es=[e for e in report['episodes'] if (e['sample_seed'] is None)==greedy and e['mode']==mode]
                n=sum(e['target_reached'] for e in es);summary[kind][('greedy' if greedy else 'sampled')+'_'+mode]=dict(success=n,total=len(es));cells.append(f'{n}/{len(es)}')
        lines.append('| '+kind+' | '+' | '.join(cells)+' |')
    lines+=['','## 判断与边界','','冻结 k2 的 CNN 加新动作探针，正常视觉贪心通过 15/17、采样通过 40/51，均高于固定画面与纯时间对照；GRU 探针对应为 16/17、39/51。视觉编码已经足以支持当前短任务，当前图传播与读出组合应成为优先排查对象。','','1. 图网络自己的 CNN 保留了明显可解码的视觉信息：k2/CNN/MLP 测试 BA 为 73.5%，固定画面 41.1%。不能再把失败简单归结为 CNN 看不懂画面。','2. 同一骨干下，256 维 motor/MLP 为 52.8%，中间节点/MLP 为 48.7%，低于 CNN；k1 也有类似下降。这支持视觉信息经当前注入和固定递归传播后变得更难读取，但有限样本和有限探针不能证明信息消失。','3. motor 探针也比原动作头有改善，因此读出尺度、优化和容量可能共同参与问题。增大到全部 motor 节点没有稳定收益，不能仅靠增加读出宽度解决。','4. 这是单个骨干种子的诊断，探针没有重训 CNN 或图网络；不能外推所有 connectome 架构。后续优先检验保持图路径的标准化读出与输入映射，并用 CNN 直连动作头作为正对照，再考虑重新接 PPO。','','## 验证','','所有骨干参数冻结且梯度为空，提取特征与部署包装器在包含历史刷新处的 logits 一致；两个闭环包装器均通过首起点逐动作的批量/顺序一致性检查。全部原始检查点 SHA-256 在运行前后相同。实验协议、节点编号、完整混淆矩阵、探针权重和逐局动作记录见 runs/layer_probes_v1/。']
    (OUT/'summary.json').write_text(json.dumps(dict(closed_loop=summary,constant=constant),indent=2)+'\n')
    (ROOT/'layer_probe_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
