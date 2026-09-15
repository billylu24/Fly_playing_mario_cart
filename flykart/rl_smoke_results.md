# 真实 MaleCNS × Mario Kart 训练 smoke test

日期：2026-09-12。**结果：PASS。真实连接图、真实 ROM、图像 CNN、循环状态、Triton 边梯度和短时 PPO 更新已端到端运行，无 OOM、NaN 或梯度断路。** 这是短时训练管线/性能检查，不是学习效果评估；完成指定更新后已停止，没有进入长训练。

## 实际用了什么

| 项目 | 本次配置 |
|---|---|
| 图来源 | 下载的 MaleCNS v1.0 原始表，选 `status == Traced` |
| 节点 / 边 | **165,122 / 25,563,197**，只保留两端均入选的真实连接 |
| 孤立节点 | 535，保留并计数，没有假设全部节点形成连通网络 |
| 初始连接权重 | 原突触数按每个目标节点的入边和归一化，使非空行和为 0.8 |
| 递质处理 | 本次未赋予递质相关正负号；初始权值为正，更新不加符号约束 |
| 输入 | 实际游戏 RGB → 84×84 灰度图；CNN 提取 64 个特征 |
| 输入映射 | 按节点索引循环分配特征到 103,270 个 ol_/visual_ 类节点；仅为 smoke 映射 |
| 输出映射 | 从 2,129 个 descending_neuron / vnc_motor / cb_motor 类节点读取活动 |
| 输出头 | 7 个动作的 categorical actor + 标量 critic |
| 可训练参数 | **25,755,453**，包括全部现存边权、CNN、actor、critic |
| 循环动态 | 每决策一步：h'=0.5h+0.5tanh(W h+视觉输入) |
| GPU / 精度 | RTX 5070，FP32，PyTorch 2.11.0+cu128 + Triton 3.6.0 |
| 进程 / 片段 | 4 个独立 Stable-Retro 进程，batch=4，展开 T=32 |
| 动作重复 | 每个决策最多执行 4 个 SNES 帧；终止则提前停止并 reset |

**这是全 Traced 集合的真实图计算测试，不是已验证的生物学输入输出模型。** 输入特征的循环分配、输出群选择、权重归一化和动力学均为明确的工程假设；本次没有证明其能实现可靠视觉—运动控制。没有把 88,384,522 个原始 segment 都当作神经元。

输入 CNN 为 Conv(1→16, kernel8,stride4) → ReLU → Conv(16→32,kernel4,stride2) → ReLU → Flatten → Linear(2592,64) → Tanh。actor 与 critic 均直接读取选定输出群。RAM 字段不输入策略，只用于 integration 的原奖励/done 和诊断。

## 实际进行了哪些训练计算

- 4 轮 rollout，每轮 4×32=128 次策略决策；总计 **512 次决策 / 2,048 个模拟器帧**。
- 每轮 2 个 recurrent PPO epoch，每个 epoch 一个完整序列 batch；共 **8 次 Adam 更新**，包括预热轮。
- 原 integration 自带的 reward 和 `isDoneTrain` 原样使用；没有把其 done 当作完赛。没有另设计任务奖励。
- gamma=0.99、GAE lambda=0.95、PPO clip=0.2、Adam lr=1e-4、eps=1e-5；梯度范数上限 0.5。
- loss=`actor_clip_loss + 0.5*(0.5*value_MSE) - 0.01*entropy`，无 value clipping。
- rollout 以 no_grad 收集，训练按原时序重放、展开反向传播；每个片段初始隐藏状态 detach。跨轮沿用数值状态，因此参数更新后存在常见的旧策略隐藏状态近似，未实现 burn-in。
- 原 integration 在本次接口只产生 terminated，脚本显式断言没有 truncated；rollout 截止正常 bootstrap。尚不支持额外 TimeLimit wrapper 的截断 bootstrap。
- 使用现有 7 个按键组合：无操作、B、B+LEFT、B+RIGHT、Y、Y+LEFT、Y+RIGHT。

## 性能实测

第 0 轮作预热；以下稳态指标来自第 1–3 轮，使用中位数。计时使用 CUDA synchronize；端到端轮次包括进程通信、图像预处理、推理、GAE、forward/backward、梯度检查和 Adam，JSON 文件写入不计入轮次耗时。

| 指标 | 实测 |
|---|---:|
| 每轮采样 128 次决策 | **0.273 s** |
| 其中策略推理 | 0.145 s |
| 其中游戏/IPC/预处理与结果传输 | 0.124 s |
| 两次 PPO epoch 更新 | **0.689 s** |
| 每轮端到端 | **0.962 s** |
| 单次 PPO 更新 | 约 **0.34 s** |
| 端到端总决策吞吐（3 轮合计） | 约 **133.5 decisions/s** |
| 端到端模拟器帧吞吐（3 轮合计） | 约 **533.9 frames/s** |
| 峰值 PyTorch 活跃张量显存 | **1,586.4 MiB ≈ 1.55 GiB** |
| allocator 保留峰值 | **1,688 MiB ≈ 1.65 GiB** |
| 建图/模型后活跃张量显存 | 885.4 MiB |
| 图/模型 GPU 初始化时间 | 约 0.97 s |
| 脚本内部建图开始至最后更新结束 | 约 5.81 s，包含 worker 启动和预热，不含 Python import、预处理与退出清理 |

四轮实际总耗时分别为 1.727 / 0.977 / 0.962 / 0.938 s。预热轮的更新约 1.12 s，稳态约 0.67–0.70 s。

显存不是整个进程 nvidia-smi 用量：不包含所有 CUDA context/库分配或其他桌面/GPU 进程。仍设置 55% PyTorch allocator cap，约 6.32 GiB；本次未接近上限。未进行长时显存泄漏、温度、功耗或独占 GPU benchmark，也未评估更长 T、更多 PPO epoch、大型 CNN 或长 rollout 的成本。

## 正确性与游戏行为证据

1. 每轮首次 PPO epoch 重放同一序列，log-prob 与采样时最大差异约 **1.19e-7**，通过 `<2e-5` 断言，说明本次顺序和初始循环状态匹配。
2. 所有更新损失有限，整体梯度裁剪检查无 NaN/Inf；encoder、边权、actor、critic 的梯度范数均存在且非零。
3. 每次更新选择一个梯度最大的边，验证 Adam 后值发生变化；本次变化约 **5.28e-5–9.83e-5**。全部边值都在可训练参数中，但不表示每条边都获得非零梯度。
4. 四轮总 reward 分别为 **0 / 0 / 20 / 10**，真实 game-state 显示部分环境 `lap:127→128`、`checkpoint:29→0`，即越过起跑线。不是完成一圈或五圈。
5. 四轮未产生终止，因此 **训练中的自动 reset / episode 边界掩码没有被这次运行实际覆盖**。worker 有终止后 reset，网络在 episode_start 清零状态，但需另做终止边界验收。此前环境测试中的 reset 已通过，不能替代训练边界测试。
6. 第二个 PPO epoch 的 KL 是小量正数，clip fraction 为 0；更新幅度很小。策略 entropy 仍约 1.946，接近 log(7)，没有证据表明策略已经学会驾驶。

因此 PASS 的含义是：**真实数据与游戏驱动下，当前配置完成了非零奖励经历、可微策略重放及实际全边权参数更新，并测得短时资源开销。** 不是任务成功或长期学习稳定的结论。

## 产物与复现

- `tests/prepare_smoke_graph.py`：按记录的规则流式读取原始连接表，准备真实 Traced 图。
- `data/malecns/traced_smoke_graph.npz`：派生图，308,262,996 字节（约 294 MiB），已 gitignore；原 Feather 未改动。
- `tests/real_smoke_graph.json`：实际节点、边数、映射、预处理耗时。扫描原始 1.52 亿行并输出派生图约 10.0 秒。
- `tests/rl_smoke.py`：限定规模/轮数的真实训练 smoke test。
- `tests/rl_smoke_runs/default.json`：逐轮指标与逐 epoch 梯度、loss、KL、参数更新证据。

在已建好的 flykart 环境中复现：

```bash
uv pip install --python .venv-bench/bin/python -r requirements.txt
.venv-bench/bin/python tests/prepare_smoke_graph.py
.venv-bench/bin/python tests/rl_smoke.py
```

算子环境及其安装说明见 `sparse_operator_results.md` / `compute_feasibility_and_plan.md`。默认只运行上述 4 轮；脚本拒绝超过 8 轮、batch>4 或 T>64 的配置，不是长训练入口。重复执行会覆盖派生图和默认日志。

没有保存这次更新后的策略 checkpoint；它只用于性能验收。4 个模拟器 worker 已退出，GPU 测试状态已释放。

## 下一步建议

当前硬件已经可以支撑**这套真实 Traced 图与轻量视觉网络的全边权短时训练**。下一步应验证终止/reset/GAE 边界、明确可靠的进度与完赛目标、改进输入输出映射，再设定单独的学习实验预算。无需为证明当前 smoke test 继续加长训练。
