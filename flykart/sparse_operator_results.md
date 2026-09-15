# 稀疏算子三条路线验证结果

> 后续更新：真实 Traced 图 + 图像 CNN + Mario Kart PPO smoke test 已通过，实测见 [rl_smoke_results.md](rl_smoke_results.md)。本文记录的是此前合成算子验证。

日期：2026-09-12。**三条路线均已测试。原生 CSR 仍有稠密化开销；自定义已有边 backward 消除了该瓶颈；Triton 融合边梯度带来了明显加速。全脑数量级的合成全边权计算已在 RTX 5070 上通过，但真实 MaleCNS + 视觉 encoder + RL 管线尚未验证。**

## 1. 对先前结论的更新

之前的 OOM 是具体原生 COO 梯度实现的限制，不是本机对“16.5 万节点”数量本身的限制。替换梯度路径之后，本次实测：

**165,122 节点，21,135,616 条边，B=4，T=32，FP32，全部现存边权可训练：峰值 PyTorch 张量显存 1,329 MiB（约 1.30 GiB），合成 forward + backward + 检查 + Adam 更新约 228 ms。**

这里采用每节点 128 条输入边的规则合成图，与实际 MaleCNS 图不同。节点数选择约 16.5 万只用于数量级探针，不代表真实 Traced 神经元集合已经转换或完整复现。两端有注释的 26,028,386 条真实边也没有在本次使用。

因此现在可以说：**“全脑数量级、全部现存边权可微更新”的计算原型已可运行。** 尚不能说真实全脑 Mario Kart RL 已能训练、显存已足够所有组件，或已经学会游戏。

## 2. 三条路线分别得到什么结果？

### 路线一：原生 CSR

保持边值为一维可训练 Parameter，使用 `sparse_csr_tensor(crow, col, values)` 和 `torch.sparse.mm`。

输出与梯度正确，比原生 COO 更节省内存、速度也更好。但 profiler 显示：乘法 backward 使用 `sparse_sampled_addmm` 后，完整梯度链中仍出现 `aten::to_dense` / `_to_dense`。N=4,096 时记录了 64 MiB 的 FP32 稠密化分配及 16 MiB 的另一稠密化分配。后者的 dtype 未单独追踪，不作过度解释。

N=32,768 时 CSR 峰值约 5.49 GiB；虽然在 55% allocator 限制下勉强通过，但这条原生“边值 Parameter → CSR 构造 → sparse.mm”的链路仍不能直接扩大全脑。此结论针对已测试链路，不能推广为所有 CSR 实现都必然稠密化。

### 路线二：只计算已有边的自定义 backward

forward 仍使用现有 CSR SpMM。对已有边 j→i 的梯度显式计算：

`grad_values[e] = sum_b(grad_output[i,b] * h[j,b])`

输入梯度通过 CSR 的转置连接计算 `W.T @ grad_output`；转置结构与索引排列提前准备，当前权值每次按该排列取值。不会调用稀疏张量构造的原生值梯度链，也不构造 N×N 稠密结果。

纯 PyTorch 原型把边分成每块 262,144 条，使用 gather、逐元素乘法与 batch 求和，临时 gather 大小受 chunk×B 限制。全边梯度本身仍需要 E 个浮点数，优化器状态也真实分配。

在 N=32,768、E=4,194,304、B=4、T=32 时，峰值约 **278 MiB**，明显低于原生 CSR 的约 **5,619 MiB**。但是 Python 分块和多次 gather 的开销使总耗时约 **269 ms**，需要进一步考虑速度。

### 路线三：Triton/CUDA 是否值得？

已实现并实测一个小型 Triton 内核：只融合已有边梯度的读取、乘法和 batch reduction；CSR forward、输入梯度 SpMM、循环状态和 Adam 均沿用 PyTorch/CUDA 库。没有重写整个模拟器或整个稀疏矩阵乘法。

同一 N=32,768、B=4、T=32 配置，总耗时从约 269 ms 降到约 **49 ms**，约 **5.5 倍**加速；全脑数量级探针从约 1,263 ms 降到约 **228 ms**，同样约 **5.5 倍**。这一配置下采用 Triton 是有实测收益的。暂时没有证据需要继续手写整套 CUDA 内核。

## 3. 正确性验证

`test_sparse_edge_ops.py` 对原生 COO、原生 CSR、PyTorch 自定义 backward、Triton 自定义 backward 做比较：

- CUDA FP32 / FP64，B=1/3/4；B=3 检查非二次幂 batch 的 mask。
- 非连续输入、非均匀入度、空行、自连接、正负权值、显式零权值。
- 小型稠密矩阵作为参考，核对 forward、所有已有边梯度和输入梯度。
- 五步共享边权循环，核对损失及跨时间累积梯度。
- 两个自定义版本均通过 FP64 finite-difference `gradcheck`。
- 纯 PyTorch 自定义路径分别验证只请求边值梯度 / 只请求输入梯度。
- 所有性能测试同时检查梯度存在、有限，encoder 与边梯度范数非零，再执行 Adam 更新。

自定义 backward 是 **一阶梯度原型**，用 `once_differentiable` 明确限制高阶梯度。当前仅支持已排序、去重、合法的固定方形图；调用方先 coalesce。没有验证图拓扑学习、高阶导数、AMP、分布式训练、torch.compile 或 checkpoint 兼容性。FP32 差异属于浮点累计顺序误差；精确数值见 correctness JSON。

## 4. 性能结果

单元格为 **峰值张量显存 MiB / 一次更新 ms**。相同配置使用相同种子、图、encoder/readout、输入和更新次数。

| N / E | B / T | 原生 COO | 原生 CSR | 自定义 PyTorch | 自定义 Triton |
|---|---|---|---|---|---|
| 8,192 / 1,048,576 | 4 / 32 | 2,245.8 / 119.7 | 457.9 / 40.4 | 89.8 / 63.6 | 82.4 / 16.9 |
| 32,768 / 4,194,304 | 1 / 8 | OOM | 5,618.7 / 64.4 | 244.3 / 7.9 | 未测 |
| 32,768 / 4,194,304 | 4 / 32 | OOM | 5,619.4 / 222.3 | 277.6 / 268.9 | 277.6 / 48.6 |
| 165,122 / 21,135,616 | 1 / 8 | 未测 | 未测 | 1,164.8 / 30.6 | 1,164.8 / 23.9 |
| 165,122 / 21,135,616 | 4 / 32 | 未测 | 未测 | 1,329.1 / 1,263.1 | 1,329.1 / 227.6 |

合计 15 个性能配置：13 个通过，2 个原生 COO 受控 OOM。全脑数量级不再测试原生 COO/CSR，因为已经观察到 N² 分配，不需要靠再次 OOM 验证已知限制。

全脑数量级 B=4/T=32，Triton 的分项中位数约为：forward 84.6 ms、backward 138.9 ms、Adam 与梯度检查 3.9 ms。allocator 保留峰值约 1,970 MiB，高于 1,329 MiB 的活跃张量峰值；两者都不等于整个进程的 nvidia-smi 用量。

### 测量口径与局限

- 环境沿用 `.venv-bench`：Python 3.12、PyTorch 2.11.0+cu128、Triton 3.6.0，RTX 5070，driver 595.84。
- FP32、每个独立子进程 warmup 一次后测三次，报告中位数；warmup 包含 Triton 首次编译，不计入稳态耗时。CUDA synchronize 包围各阶段。
- 使用小型 Linear(64,128) encoder 与 Linear(128,8) readout，随机输入与平方损失；每步循环更新一次，所有边权参与求导和优化器更新。结构性零梯度的边仍可能存在，不声称所有神经元都有丰富活动。
- 没有游戏采样、CNN、PPO loss、价值网络、长 rollout、视频缓存或混合精度。不是游戏帧吞吐 benchmark，也不能由 228 ms 推出学会驾驶需要几小时。
- 指标为 graph 初始化之后的 PyTorch allocator 峰值，包含常驻图/参数/梯度/Adam，但不统计初始化过程的峰值、CUDA context、库内部非 allocator 分配及其他进程。初始化本身在预算内完成。
- 55% allocator cap（约 6.32 GiB）保持不变；不终止桌面或其他用户 GPU 进程。全脑合成试验有余量，但实际可用内存随其他进程变化。
- 规则 128 入度合成图的内存访问比真实非均匀神经连接更规则。本次不能确认真实结构、输入输出群和活跃神经元分布下的速度。
- Graph 为公平比较提前保存了正向、转置及 COO 索引，原生路径也保留这部分结构。这使原生测试有少量未用索引常驻，但不足以解释 GiB 级稠密临时分配差异。

## 5. 接下来可以做什么？

1. **优先将 Triton 版本作为下一阶段算子候选，保留纯 PyTorch 自定义版本作为参考。** 不必继续沿当前原生 COO 全边梯度路径扩规模。
2. 确定真实神经元筛选规则和真实边数；在真实子图上重复正确性与资源验收，再逐步扩大到最终目标集合。当前数据仍未转换。
3. 加入实际图像 encoder、读出/价值头和预期 rollout 缓存，先固定录制画面做 forward/backward 显存验收。先 B=1/T=8，再 B=4/T=32，记录整卡占用并保留余量。
4. 只有以上通过后，再决定是否用全脑全边权做 RL；固定连接、gain 与全边权仍应保留为实验对照。算子能训练不意味着全边训练在学习效果或生物学解释上一定更好。

**更新后的建议：全脑全边权不再被这次 N² 算子瓶颈直接否决，可以进入真实数据的计算验收阶段；尚未达到直接启动完整 RL 的验证程度。**

## 6. 文件与复现

从 flykart 目录运行（已有 benchmark 环境，无需另装 CUDA）：

```bash
.venv-bench/bin/python tests/test_sparse_edge_ops.py
.venv-bench/bin/python tests/bench_sparse_operators.py
.venv-bench/bin/python tests/bench_sparse_operators.py --extra
.venv-bench/bin/python tests/profile_sparse_alternatives.py
```

- `tests/sparse_edge_ops.py`：CSR 图结构及已有边自定义 backward。
- `tests/sparse_edge_triton.py`：融合的已有边梯度内核。
- `tests/sparse_operator_correctness.json`：正确性结果。
- `tests/sparse_operator_benchmark.json`：COO/CSR/PyTorch 自定义对比。
- `tests/sparse_operator_extended.json`：Triton 与全脑数量级探针。
- `tests/sparse_alternatives_profile.json`：CSR 稠密化及自定义路径 profiler 证据。

没有改动 MaleCNS Feather、ROM 或游戏 integration，没有开始 RL 训练。
