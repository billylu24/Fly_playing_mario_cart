"""Audit local recovery data and render numerical-only coverage report."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/recovery_demonstrations_v1'


def read(name):return json.loads((OUT/name).read_text())

def run():
    status=read('status.json')
    assert status['state']=='completed' and status['sources_unchanged']
    records=read('branches.json');baselines=read('baselines.json');selected=read('selected_manifest.json')
    original=json.loads((ROOT/'runs/supervised_architecture_v1/manifest.json').read_text())
    train={r['start_id'] for r in original if r['split']=='train' and r['accepted']}
    heldout={r['start_id'] for r in original if r['split']!='train'}
    assert len(baselines)==3*len(train)==129
    assert {r['start_id'] for r in baselines}==train
    assert not {r['start_id'] for r in records}&heldout
    assert len({(r['seed'],r['start_id']) for r in baselines})==129
    baseline_map={(r['seed'],r['start_id']):r for r in baselines}
    paired=Counter();selected_keys=set();fingerprints=Counter();deduplicated=[];counted=np.zeros(3,dtype=np.int64)
    for r in records:
        assert r['prefix_and_takeover_match']
        assert r['accepted']==bool(r['target_reached'] and r['jumps']==0)
        b=baseline_map[(r['seed'],r['start_id'])]
        assert r['actions'][:r['takeover']]==b['actions'][:r['takeover']]
        assert r['takeover_observation_sha256']==b['observation_hashes'][r['takeover']]
    for b in baselines:
        for t in [8,32]:
            rr=[r for r in records if r['seed']==b['seed'] and r['start_id']==b['start_id'] and r['takeover']==t]
            if len(b['actions'])<=t:
                assert not rr
                continue
            assert {r['teacher'] for r in rr}=={'cnn','waypoint'} and len(rr)==2
            if not b['target_reached']:
                ok={r['teacher'] for r in rr if r['accepted']}
                paired['both' if len(ok)==2 else next(iter(ok))+'_only' if ok else 'neither']+=1
    for r in selected:
        key=(r['seed'],r['start_id'],r['takeover'])
        assert key not in selected_keys
        selected_keys.add(key)
        assert r['accepted'] and not r['baseline_success']
        alternatives=[x for x in records if (x['seed'],x['start_id'],x['takeover'])==key and x['accepted']]
        assert r['teacher']==('waypoint' if any(x['teacher']=='waypoint' for x in alternatives) else 'cnn')
        p=OUT/'data'/r['filename']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        with np.load(p) as d:
            x,y,a=d['observations'],d['actions'],d['executed_actions']
            assert x.dtype==np.uint8 and x.shape==(len(a),84,84)
            assert len(y)==len(a)==len(r['actions'])
            assert np.array_equal(a,r['actions'])
            assert np.all(y[:r['takeover']]==-100)
            assert np.array_equal(y[r['takeover']:],a[r['takeover']:])
            assert np.array_equal(np.bincount(y[y>=0],minlength=3),r['action_counts'])
            for t in range(r['takeover']+1):
                assert hashlib.sha256(x[t].tobytes()).hexdigest()==baseline_map[key[:2]]['observation_hashes'][t]
            fingerprint=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
            if fingerprint not in fingerprints:deduplicated.append(dict(r,sequence_sha256=fingerprint))
            fingerprints[fingerprint]+=1
            counted+=np.bincount(y[y>=0],minlength=3)
    assert len(selected)==paired['both']+paired['cnn_only']+paired['waypoint_only']
    (OUT/'training_manifest.json').write_text(json.dumps(deduplicated,indent=2)+'\n')
    gate=read('quality_gate.json')
    assert counted.tolist()==gate['action_counts']
    audit=dict(train_only=True,all_prefixes_match=True,all_selected_frame_hashes_and_masks_match=True,
        matched_failure_branch_pairs=dict(paired),unique_full_selected_trajectories=len(fingerprints),
        exact_duplicate_selected_trajectories=len(selected)-len(fingerprints),supervised_action_counts=counted.tolist())
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    summary=read('summary.json')
    lines=['# 训练起点上的恢复示范验证', '',
        f"完成 {len(baselines)} 局 fly 自主基线和 {len(records)} 局控制器接管，总计 {len(baselines)+len(records)} 局。使用全部 43 个合格原训练起点和三个已训练的 joint 种子；没有训练新模型。", '',
        '本轮只验证恢复数据是否可用，不是自主 fly 的改进结果。三个种子共用此前的 CNN、教师和预训练起点，只改变此前训练采样顺序。', '',
        '## 固定方案', '',
        '学生驾驶 8 或 32 次决策后，由直接 CNN 或现有 waypoint 控制器驾驶到结束。两种教师从相同实际状态接管；原任务时间和停滞计时不重置。waypoint 在学生前缀期间持续更新，避免接管时错误地回到第一个路标。RAM 只供示范控制器使用。', '',
        '基线提前结束则跳过相应分支。只将基线失败、且接管后达到目标并无 checkpoint 跳跃的轨迹作为恢复训练候选；同一 seed/起点/接管时刻优先选成功 waypoint，否则选成功 CNN。接管前完整图像历史保留，但动作监督标为 -100。', '',
        '## 自主基线', '', '| seed | 成功/起点 |', '|---:|---:|']
    for seed in range(3):
        rr=[r for r in baselines if r['seed']==seed]
        lines.append(f"| {seed} | {sum(r['target_reached'] for r in rr)}/{len(rr)} |")
    lines += ['', '## 基线失败时的接管结果', '', '| 接管教师 | 第 8 次决策接管 | 第 32 次决策接管 |', '|---|---:|---:|']
    for teacher in ['cnn','waypoint']:
        cells=[]
        for t in [8,32]:
            r=summary['baseline_failures'][f'{teacher}_{t}'];cells.append(f"{r['success']}/{r['total']}")
        lines.append(f"| {teacher} | "+' | '.join(cells)+' |')
    lines += ['',f"配对比较（同一失败起点、种子、接管时刻）：双方成功 {paired['both']}，仅 CNN 成功 {paired['cnn_only']}，仅 waypoint 成功 {paired['waypoint_only']}，双方失败 {paired['neither']}。这些分支共享起点与历史，不作为独立统计样本。", '',
        '## 可用数据与限制', '',
        f"按预先规则选出 {len(selected)} 条恢复轨迹，覆盖 {gate['distinct_starts']} 个不同训练起点、prefix {gate['prefix_frames']}；三类动作计数（直行/左/右）为 {counted.tolist()}。完整观察和监督序列精确去重后 {len(fingerprints)} 条；不同序列仍可能高度相关。", '',
        f"预先声明的覆盖门槛{'通过' if gate['passed'] else '未通过'}：至少 12 个不同起点、至少 3 个 prefix、三类动作均存在。该门槛只检验覆盖，不证明最优动作或泛化。", '',
        '成功筛选会偏向可恢复状态；失败分支被完整记入数值报告，不能假定其恢复需求已经解决。waypoint 利用位置/方向等特权信息，部署时 fly 仍只能看图像，能否模仿成功必须另做闭环验证。CNN 成功轨迹也只说明整段控制成功，不保证每个动作最优。', '',
        '## 下一轮训练设计', '',
        '若门槛通过，下一轮采用三组：A 旧数据 MSE+教师 KL；B 在 A 的一半批次槽位使用旧示范动作 CE；C 与 B 相同的损失和批次比例，但该槽位换为恢复示范。B/C 用于隔离恢复数据的作用，避免把动作损失变化误判为数据收益。各组保持相同 warm-start、更新数、学习率、冻结 CNN/动作头、图拓扑和动力学。训练前锁定精确权重、采样计划和新评估起点；只能用验证集选择模型。', '',
        '恢复数据按完整轨迹处理、保留前缀，并先验证监督掩码和 Context 重建一致。各采样种子都要评估正常/冻结图像/时间控制以及直接 CNN；不得把教师接管的成功率当作 fly 自主成功率。暂不增加抑制机制或延长 RL。', '',
        '## 校验与文件', '',
        '训练起点隔离、配对前缀与接管画面哈希、所有选中 NPZ 文件哈希、监督掩码和动作计数检查通过。来源哈希保持不变。`training_manifest.json` 保留 74 条精确去重轨迹，原 81 条选择结果仍完整保留。', '',
        '另用 `tests/audit_recovery_overlap.py` 检查原验证/测试和四轮归档新起点示范：恢复监督帧及其前缀与这些示范的精确画面重叠均为 0；这不表示场景或赛道独立。', '',
        '代码：`training/recovery_demonstrations.py`；方案、数值结果与数据索引：`runs/recovery_demonstrations_v1/`。包含游戏画面的 `data/*.npz` 仅留本地，不纳入 Git。', '']
    (ROOT/'recovery_demonstrations_results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=status,gate=gate,audit=audit),indent=2))

if __name__=='__main__':run()
