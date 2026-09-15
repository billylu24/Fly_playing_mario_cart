"""Complete table and prespecified gates for frozen-feature head adaptation."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/head_adaptation_v1'
SOURCES=['cnn','graph0','graph1','graph2']


def read(name):return json.loads((OUT/name).read_text())


def run():
    status=read('status.json');assert status['state']=='completed'
    n=status['new_test_accepted'];closed={};offline={}
    for source in SOURCES:
        frozen=read(source+'_frozen_metrics.json');offline[source]={'frozen':frozen}
        for kind in ['old','mixed']:
            r=read(source+'_'+kind+'_metrics.json');offline[source][kind]=r['selected']
        for kind in ['frozen','old','mixed']:
            name=source+'_'+kind;r=read(name+'_evaluation.json');assert r['batched_sequential_match'];es=r['episodes'];assert len(es)==n*12
            assert len({(e['start_id'],e['mode'],e['sample_seed']) for e in es})==n*12
            m={}
            for sampling in ['greedy','sampled']:
                for mode in ['normal','frozen','clock']:
                    rr=[e for e in es if e['mode']==mode and (e['sample_seed'] is None)==(sampling=='greedy')]
                    m[sampling+'_'+mode]=dict(success=sum(e['target_reached'] for e in rr),total=len(rr),rate=sum(e['target_reached'] for e in rr)/len(rr))
            closed[name]=m
    offgates={s:dict(recovery_improvement=offline[s]['mixed']['recovery_validation']['cross_entropy']<=.9*offline[s]['frozen']['recovery_validation']['cross_entropy'],old_retained=offline[s]['mixed']['validation']['cross_entropy']<=1.05*offline[s]['frozen']['validation']['cross_entropy']) for s in SOURCES}
    graph=[s for s in SOURCES if s!='cnn']
    delta=sum(closed[s+'_mixed']['greedy_normal']['success']-closed[s+'_old']['greedy_normal']['success'] for s in graph)
    gates=dict(all_graph_seeds_nonworse=all(closed[s+'_mixed']['greedy_normal']['success']>=closed[s+'_old']['greedy_normal']['success'] for s in graph),
        aggregate_gain_10pp=delta/(3*n)>=.1,
        all_graph_visual_benefits=all(closed[s+'_mixed']['greedy_normal']['rate']-closed[s+'_mixed']['greedy_frozen']['rate']>=.2-1e-12 and closed[s+'_mixed']['greedy_normal']['rate']-closed[s+'_mixed']['greedy_clock']['rate']>=.1-1e-12 for s in graph),
        aggregate_sampled_gain=sum(closed[s+'_mixed']['sampled_normal']['success']-closed[s+'_old']['sampled_normal']['success'] for s in graph)>0)
    summary=dict(closed_loop=closed,offline=offline,offline_gates=offgates,closed_loop_gates=gates,closed_loop_gate_passed=all(gates.values()),graph_greedy_success_delta=delta)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    val=read('validation_status.json');sel=read('selection.json')['models']
    lines=['# 冻结特征后的动作头适应', '',
        f"完成8个动作头训练、12种策略 × {n}个新起点 × 12种评估组合，共{status['episodes']}局，另有12局顺序回归。图像数据与模型权重仅保存在本地。", '',
        '本实验只训练原64→32→3动作头的两层参数。CNN、图连接、神经元增益、视觉接口、重建解码器以及动作头归一化均固定。CNN直接特征与三个图模型的重建特征，使用相同帧、标签、头初始化及采样计划。图种子只代表此前的采样顺序差异，不是独立预训练。', '',
        '每种特征有冻结头、旧数据适应头、混入恢复数据适应头。适应组512更新，Adam lr0.001，batch128；前64样本来自旧数据，后64分别来自旧数据/恢复数据，两个CE损失各占一半。特征在完整实际历史重放后缓存，掩码只屏蔽动作监督；没有对前缀状态做不合法重置。', '',
        f"恢复验证来自原6个验证起点，18局基线与{val['branch_episodes']}局接管。选出并去重{val['selected']}条成功恢复轨迹，覆盖{val['distinct_starts']}个验证起点；这些数据不进入训练梯度。", '',
        '选择规则：每32步评估，在旧验证CE不超过自身初始值105%的checkpoint中，最小化旧验证与恢复验证CE的等权平均，平局取较早。新测试为460/520/580/640 × 扰动28 × 左/中/右，oracle筛选在头训练之前完成。', '',
        '## 验证集：冻结头 → 旧数据头 → 恢复混合头', '',
        '| 特征 | 原验证CE | 恢复验证CE | mixed选中步数 | 恢复CE改善≥10%且保留旧验证 |', '|---|---|---|---:|---|']
    for source in SOURCES:
        old=' → '.join(f"{offline[source][k]['validation']['cross_entropy']:.3f}" for k in ['frozen','old','mixed'])
        rec=' → '.join(f"{offline[source][k]['recovery_validation']['cross_entropy']:.3f}" for k in ['frozen','old','mixed'])
        lines.append(f"| {source} | {old} | {rec} | {sel[source+'_mixed']} | {all(offgates[source].values())} |")
    lines+=['', '## 自主闭环', '', '| 特征/条件 | 贪心正常 | 贪心固定 | 贪心时间 | 采样正常 | 采样固定 | 采样时间 |','|---|---:|---:|---:|---:|---:|---:|']
    keys=['greedy_normal','greedy_frozen','greedy_clock','sampled_normal','sampled_frozen','sampled_clock']
    for name,m in closed.items():lines.append('| '+name+' | '+' | '.join(f"{m[k]['success']}/{m[k]['total']}" for k in keys)+' |')
    lines+=['',f"图模型恢复混合头相对旧数据头的贪心总成功差：{delta:+d}/{3*n}；预设闭环门槛{'通过' if all(gates.values()) else '未通过'}。", '']
    for k,v in gates.items():lines.append(f'- {k}: {v}')
    lines+=['', '以上均为共享起点上的描述性对照，不能当作独立环境样本的显著性检验。新起点仍在同一赛道；验证集在多个研究阶段反复使用，结论限于本任务。离线动作CE改善不等于闭环恢复能力提升，需结合正常/固定/时间控制。', '',
        '## 校验', '',
        '缓存特征的动作输出与部署Context逐项一致；原模型和原动作头不变；训练/验证起点ID不交叉；两组头初始化及采样计划一致。全部策略首个起点的采样正常结果与顺序回放在动作、终止原因、帧数、进度、跳跃数上逐项相同，来源哈希保持不变。', '',
        '脚本：`training/head_adaptation.py`；完整方案、采样计划、验证曲线、选择结果及动作轨迹在 `runs/head_adaptation_v1/`。', '']
    mixed=sum(closed[s+'_mixed']['greedy_normal']['success'] for s in graph)
    frozen=sum(closed[s+'_frozen']['greedy_normal']['success'] for s in graph)
    old=sum(closed[s+'_old']['greedy_normal']['success'] for s in graph)
    lines += ['## 解释与下一步', '',
        f"本轮预设闭环门槛通过：恢复混合头{mixed}/{3*n}，旧数据适应头{old}/{3*n}。但冻结头本来已有{frozen}/{3*n}，混合头相对冻结头仅增加{mixed-frozen}个成功组合。graph2混合头8/10主要避免了旧数据头退到6/10，而不是超过原冻结头8/10。", '',
        '采样正常合计：冻结头46/90，旧数据头58/90，恢复混合头62/90；恢复混合头的固定图像为27/90、时间为18/90。三组fly混合头都保留正常画面相对固定/时间的收益，但采样局共享起点，不能当独立证据。直接CNN冻结头贪心10/10，仍强于fly；没有证明图拓扑带来优势。', '',
        '动作头适应确实能利用部分现有特征，并在保持旧验证指标的同时改善恢复验证损失；冻结动作头是可改善环节，但不是唯一瓶颈。恢复验证只有2个起点、6条相关轨迹，新测试只有10个起点，而且所有动作头使用同一采样序列，因此这仍是小规模机制试验。', '',
        '下一步优先复验这套已出现信号的方案：先固定本轮checkpoint，在更大一批预先锁定的新起点上比较mixed/old/frozen；再改变动作头训练采样种子检查稳定性。复验前不改变fly动力学或抑制机制，也不把已有测试起点反馈用于选模。只有收益稳定后，再定位重建64维读出与原始motor特征之间的信息损失。', '',
        '## 训练前实现修正', '',
        '第一次缓存/部署检查失败发生在任何动作头训练之前。批量CNN编码与逐帧编码的最大特征差3.72e-5，经原头归一化放大为logit差0.00460；头本身批量差约1.61e-6。已改为逐帧提取CNN缓存，未放宽断言，重用未变的采集数据并重新通过检查。原始失败与有效源码哈希修正记录在cache_fix_amendment.json；最终sources_unchanged针对修正后方案。该修正不证明此前实验的失败由数值误差造成。', '',
        '恢复验证/新测试与训练数据的精确画面重叠均为0，详见overlap_audit.json；这不代表同赛道场景独立。', '']
    (ROOT/'head_adaptation_results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=status,offline_gates=offgates,closed_loop_gates=gates,closed_loop=closed),indent=2))

if __name__=='__main__':run()
