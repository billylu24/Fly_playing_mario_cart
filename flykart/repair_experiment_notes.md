# PPO/BPTT 修复与小预算验证

## 修改范围

保留固定 MaleCNS 连接矩阵、原始视觉输入映射和脑输出读出，不开放全边训练，不加入绕过脑网络的视觉直连。原始五圈任务、七动作、repeat=4 仍是默认选项。

### 已实现

1. **消除跨参数版本隐藏状态缓存**：每 32 步从零状态重放最近 64 个观测（含 episode reset mask）；行为采样和 PPO 更新使用同样上下文。各 PPO epoch 在当前参数下重新生成上下文，更新前检查 log-prob 误差 <2e-5。bootstrap 也按下一采样块的重建规则计算。
   - 这是明确的有限上下文策略，**不是精确重放整局历史**。64 步 burn-in 不求梯度，BPTT 仍为 32。
   - 以额外前向计算换取一致性；不能与旧配置的吞吐直接比较。评估也每 32 步刷新上下文；episode 开始清零。
   - 旧状态策略 checkpoint 不允许直接 resume；旧文件未改动。
2. **短任务课程参数**：`--target N`（1–150），首次依序达到目标时结束；短任务终点奖励为 `5*N/150`。只有完整 150 checkpoint 才标记五圈 success。
3. **无进展停止**：`--stall-frames N`，连续 N 游戏帧未推进 frontier 就终止，减少长时间卡住。这里定义为有限短任务的 terminal，不是遗漏 bootstrap 的 time-limit truncation。其任务价值与原始五圈不同。
4. **动作消融**：`--action-set drive` 仅保留油门/油门左/油门右；`--repeat` 控制持续帧数。三动作模式是起步课程，不宣称适合完整比赛。
5. **冻结内部参数对照**：`--freeze-gain`，只更新 encoder/actor/critic。
6. **诊断与评估**：记录原始优势均值/标准差、正奖励比例、encoder/gain 梯度；增加随机、固定油门、固定左右转向、初始/最终模型对照脚本，独立使用 2000 起的评估种子。gamma、GAE lambda、熵系数可配置。
7. **时间惩罚消融**：`--time-penalty 0` 可取消时间惩罚。无进展提前终止加负时间奖励有偏好提前失败的风险，短课程下一轮应单独比较取消惩罚；默认值不变。本次 4096 步仍使用原值，未将后加的选项混称为已验证的学习收益。

### 不作“已经修好”的承诺

视觉反应弱、gain 学习贡献小，仍是学习层面的待验证问题。没有用辅助标签或跳过 connectome 的通路把问题掩盖掉，也没有因为短任务获得奖励就认定模型看懂赛道。下一步必须验证超过固定动作对照，以及连续冻结画面的消融成绩。

## 验证命令

在 `flykart/` 目录：

```bash
.venv-bench/bin/python tests/test_baseline.py
.venv-bench/bin/python tests/test_repairs.py
.venv-bench/bin/python -m training.train_gain --run runs/gain_repair_smoke_v2 --budget 4096 --envs 4 --sequences 4 --target 3 --stall-frames 600 --action-set drive --repeat 8 --eval-every 4096 --eval-episodes 2
.venv-bench/bin/python tests/evaluate_repair_controls.py --run runs/gain_repair_smoke_v2 --episodes 10
```

测试覆盖原有五圈进度、reset、GAE、梯度及 checkpoint；新增课程终点、stall 计时重置、非整除动作 repeat 的终止帧数、冻结 gain、参数改变后的上下文重建。CUDA 稀疏浮点累加存在约 3e-8 的重复运行误差，测试使用相应浮点容差而非逐位相等。

## 后续实验门槛

本次混合配置只作 smoke test，不用于分别归因每项修改。后续固定任务和交互预算，以单变量方式比较状态修复、动作配置、gain 冻结，并分别报告决策数和游戏帧数（repeat 改变后两者不等价）。至少跨 3 个训练 seed，优于随机/固定动作后，再进入更长路段与一圈任务；目前不自动开始长训练。

## 实测结果

`runs/gain_repair_smoke_v2`：4096 决策、32517 游戏帧、128 次更新，约 101 秒，4 环境中位 41.1 决策/秒，峰值 allocated 934 MiB。较慢包含更长上下文重算、并行数降低、动作 repeat 增大等因素，不作同条件吞吐比较。

- 更新前 log-prob 重放偏差最大 2.98e-7；原实验曾为 0.2088。这是状态一致性修复的直接验证。
- 32 个 rollout 中 1 个无正奖励（3.1%）；原实验 82.4%。任务、批大小、动作配置均变了，不能只归因于状态修复。
- 完成 62 个短任务 episode，35 次达到目标；encoder 和 gain 梯度均有限非零，W 保持不变。
- 独立评估 seeds=2000..2009，目标仅前 3 个 checkpoint：

| 策略 | 达到短目标 | 平均进度 |
|---|---:|---:|
| 随机三动作 | 0/10 | 2.0 |
| 固定油门 | 0/10 | 2.0 |
| 固定油门左 | 0/10 | 1.0 |
| 固定油门右 | 0/10 | 0.0 |
| 初始模型 | 3/10 | 2.2 |
| 4096 步模型 | 8/10 | 2.5 |

固定动作在同一确定性初始状态下重复，10 次不是 10 个不同赛道；随机对照与模型采样使用不同 RNG，不能把同 seed 解读为相同动作序列。结果提供短任务改善的初步信号，不能证明视觉依赖、gain 的独立贡献或泛化。未跑完一圈。

实现迭代说明：4096 步验证包含下一采样块的 bootstrap 状态刷新和短目标 best-checkpoint 排序；其后补充可选时间惩罚，run 的 `config.json` 记录启动时源码 hash。随后 `runs/frozen_repair_gate_v3` 完成 512 步当前最终实现验证（冻结 gain、零时间惩罚）：16 次更新、约 12.2 秒，gain 始终为 1，W 未变化，无数值错误。两次结果不是同配置消融。

原有 `gain_seed0_v1` checkpoint 未覆盖。完整原始数值见各 run 的 `metrics.jsonl`、`status.json` 和 `control_evaluations.json`。
