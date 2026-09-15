# FlyKart PPO＋截断 BPTT 基线实验方案 v1

> 执行更新：用户已授权启动。验收通过后，按性能探针选16环境/H32/M16/T32，seed=0的200k决策实验已启动。完整配置、偏差与证据见 [gain_experiment_start.md](gain_experiment_start.md)。下面保留原始方案，其中“本次不启动”指方案设计时。

状态：设计稿已定出可执行默认值；**本次不启动训练**。日期：2026-09-12。

第一实验：**真实 MaleCNS Traced 图，固定连接权重，仅训练每神经元 gain、视觉 encoder、actor、critic。** 先检验能否学会固定赛道驾驶，再运行冻结 gain、普通 GRU 和随机重连对照。全边权训练作为后续扩展，不能替代第一实验。

## 1. 研究问题与结论边界

主问题：固定果蝇连接结构配合有限的神经元增益适应，能否通过 PPO/BPTT 学会视觉驾驶？

- H1：gain 组可以在固定 Mario Circuit Time Trial 完成五圈。
- H2：gain 组相对完全冻结脑内参数组提高完赛率或样本效率。
- H3：真实图 gain 组相对保留度数的随机重连图 gain 组有优势。

H1 成功不能独立证明 H2/H3，也不能证明真实果蝇的局部学习机制得到复现。模型是连接组约束的人工循环策略。参数数目、节点数和游戏表现均不能单独说明生物学真实性。

现有证据：真实图全边权 smoke 已通过，B=4/T=32 时张量峰值约 1.55 GiB，约 133.5 策略决策/秒。但只有 512 次决策，未覆盖 episode 终止，未学会驾驶，不能视为 gain 组的性能或学习结果。

## 2. 数据与模型冻结项

### 2.1 节点、边和权重

- 从指定 MaleCNS v1.0 annotation 选择 `status == Traced`，按 bodyId 升序建立连续索引。
- N=165,122；仅保留两个端点都入选的边，E=25,563,197。535 个孤立节点保留并报告。
- 方向严格为 `W[target,source]`，以 `body_pre` 为 source、`body_post` 为 target。
- 初始 `W_ij = 0.8 * count_ij / sum_k count_ik`，无入边的行全零。
- 第一组 W 固定且不进入 optimizer；不会优化原始突触数量或修改 Feather 文件。
- 本轮不按递质分配兴奋/抑制符号。全正、入行归一化是工程假设，不是生理效能估计；各连接组对照一致使用，局限必须报告。
- 缓存固定 CSR 的 W 和 Wᵀ。gain 模式不需要边权梯度，也不运行 Triton 边梯度内核；仍必须让梯度经过 W 回到 encoder。

复用现有派生图可保证数据一致；原始文件哈希见 data/malecns/README.md。每个 run 记录派生图 SHA-256、节点列表/掩码哈希、代码版本、integration 提交和 ROM SHA-1。

### 2.2 视觉输入和输出

第一版本明确沿用当前可运行映射，避免同时更改太多因素：

1. 全画面（包含上半驾驶视角和下半赛道地图）转灰度、缩放 84×84，uint8 缓存，送入 CNN 时除以 255。无 frame stack；记忆由循环活动承担。
2. CNN：Conv 1→16,8×8,stride4 → ReLU → Conv 16→32,4×4,stride2 → ReLU → Flatten → Linear 2592→64 → Tanh。
3. 103,270 个 `ol_` 或 `visual_` superclass 节点接受输入。节点索引 i 接收 `z[i mod 64]`，其余节点输入为零。
4. 从 2,129 个 `descending_neuron/vnc_motor/cb_motor` 节点读取活动，分别送入 Linear(2129,7) actor 和 Linear(2129,1) critic。
5. actor 的 observation 仅来自画面；RAM、checkpoint、reward、真实神经元标签和成功标志均不能作为额外策略输入。

这不是视网膜拓扑或可信的生物感觉编码，bodyId 索引循环映射也没有生物意义。其作用是建立一个固定、可复现的工程基线。后续改输入映射必须建立新版本并重跑相关对照，不能把改进后结果混入 v1。

### 2.3 动力学和训练参数

每次策略决策执行一次网络更新，所有单位为模型步，暂不对应毫秒级生理时间：

`g_i = 0.5 + sigmoid(a_i)`，`a_i=0` 初始化，所以 g 初始为 1，范围 (0.5,1.5)。

`h_next = 0.5*h + 0.5*tanh(g ⊙ (W@h) + I)`。

gain 调制循环输入，不调制直接视觉输入；位置不能随实验组随意改变。h 初始全零，episode 结束后的新 episode 清零。actor/critic 读取 h_next。

| 参数组 | GAIN 主组 | FROZEN 对照 |
|---|---|---|
| CNN encoder | 可训练 | 可训练 |
| a（每节点一维） | 可训练，165,122 个 | 固定为 0 |
| W | 固定 | 固定 |
| actor/critic | 可训练 | 可训练 |

同现有 CNN/head 结构计算：FROZEN 约 192,256 个可训练参数，GAIN 约 **357,378**；应由代码逐组打印并断言实际计数。连接图占内存，但固定权值不需要 Adam 状态。

不施加 gain 正则项作为第一版默认，记录其分布、接近上下界比例、类型分组统计。不能把 gain 边界当作“模型一定稳定”的证明；原始 W 行和 0.8 乘最大 gain 可达到 1.2。

## 3. 任务、动作和 episode

任务：固定 Super Mario Kart USA ROM、Mario、Mario Circuit 的 Time Trial 存档，从统一初始状态在 **300 秒模拟时间**内按合法 checkpoint 顺序完成五圈。

- ROM SHA-1：47e103d8398cf5b7cbb42b95df3a3c270691163b。
- 使用现有 MarioCircuit_M.state；检查其赛道/角色/模式后冻结存档哈希。
- 每决策 action repeat=4，重复期间逐 SNES 帧检查进度、成功及失败，遇到终止立即停止重复。
- 7 个离散动作：无操作、B、B+LEFT、B+RIGHT、Y、Y+LEFT、Y+RIGHT。先确认 Y 的刹车效果；失败则修正动作表并冻结新版本后才能训练。
- 跳跃、漂移、菜单按键不纳入 v1。该动作集合能否完成赛道须有人工或脚本驾驶可达性证据，不因“能 step”就假设可以完赛。
- 时间从 reset 对应模拟帧开始计算，不用真实 wall-clock；按约 60 fps 上限为 18,000 模拟帧，启动前确认 core 标称帧率。最多约 4,500 次决策，终止帧不对齐时实际数略变。

### 明确区分终止原因

1. **success**：官方游戏状态/画面表明完赛，并与五圈合法进度一致；`lap-128>=5` 仅作为待验证候选，不是当前已验证实现。
2. **failure**：意外离开比赛模式等经验证的不可继续状态，或超过有限时长预算。
3. **collision/off-road**：可恢复的驾驶事件，记录但不自动结束 episode。

正式 wrapper 不使用上游 `isDoneTrain` 的“撞墙即 done”作为任务终止，也不把底层 done 当成功。应提供专用 scenario，只暴露 RAM 信息，由 wrapper 独立管理任务；上游文件保留原样。

300 秒上限定义为本任务自身的有限时域终点：超时记 `terminated=True`、失败、bootstrap=0。它不是普通 Gym TimeLimit 的 `truncated`。如果以后改成外部截断，必须改变 GAE 处理，使用截断前最后画面 bootstrap；不能从 auto-reset 后画面 bootstrap。异常/用户中断不作为正常训练样本。

## 4. 进度与 reward v1

reward 是工程目标，不解释为多巴胺或生物局部学习信号。只使用少量项，暂不加速度奖励、碰撞罚分或逆行罚分。

`r = ΔF / C + 5 * first_success - 0.001 * (actual_frames / 4)`。

- C 为每圈经过的合法 checkpoint 段数，当前观察值为 30，须以完赛轨迹验证。
- F 为本 episode 到达的**最高合法进度段数**，范围 0..5C。已经到过的区域反复跨线不再提供正奖励。
- 第一回合法越过起跑线时设置 started，进度基线 F=0，**不给起跑线奖励**。不能把初始 raw lap=127、checkpoint=29 直接当作已完成进度。
- 之后逐帧记录当前合法位置与历史 frontier。只有从合法前一段进入后继段才推进路径计数，正确处理 29→0 回绕和圈数变化；从落水恢复/重新定位造成的跳变先分类，不自动接受为进度。
- 后退后再次驶过旧段只恢复当前位置，不增加 F；必须重新到达 frontier 之后才有新奖励。进度事件检测需要真实轨迹验证，不能直接累加 `max(0, checkpoint_delta)`。
- 若每帧可能跳过多个 checkpoint，只有经过实测认可的跨段规则才能补计；规则不明先报诊断，不随意奖励。
- first_success 仅发放一次，成功事件结束 episode。超时/离开比赛没有额外固定罚分，已承担耗时成本与失去成功奖励。

从未经过的新 checkpoint 贡献约 1/30；一圈总进度奖励约 1，五圈约 5，加完赛奖励 5；300 秒满时预算的耗时成本约 4.5。此尺度是 v1 预先选择的目标，不表示已调到最优。撞墙和离路通过失去进度及耗时体现成本。

阶段性“一圈成功”只用于日志/门槛，**不提前终止或额外奖励**，避免改变五圈任务。

## 5. PPO 与截断 BPTT 的完整默认配置

| 项目 | 默认值 |
|---|---|
| rollout 环境数 P | 4，先采用已运行进程数 |
| 每环境 rollout 长度 H | 128 次决策 |
| 每次采样总量 | P×H=512 transitions |
| BPTT 学习片段 T | 32 次决策 |
| 一个训练 minibatch | 4 条连续片段，即 128 个有效 transitions |
| 一个 rollout 的片段数 | 16；每 epoch 共 4 个 minibatch |
| PPO epochs | 4，因此每轮采样最多 16 次 optimizer step |
| gamma / GAE lambda | 0.99 / 0.95，按决策步计 |
| PPO clip | 0.2 |
| actor/critic CNN optimizer | Adam lr=1e-4、eps=1e-5、weight_decay=0 |
| gain optimizer 参数组 | 同一 Adam，lr=1e-4，a 可训练 |
| entropy coefficient | 0.01，固定 |
| value coefficient | 0.5，应用于 0.5×MSE |
| gradient clipping | 所有可训练参数联合 global L2 norm=0.5 |
| target KL | epoch 的采样加权平均 approximate KL >0.02 则停止该轮余下 epoch |
| advantage normalization | 全 rollout 有效 transitions 上归一化，不对 padding 计数 |
| precision | FP32；首轮不用 AMP/compile |
| reward normalization | 不使用；明确记录 raw reward |
| value clipping / lr schedule | 均关闭，减少第一版变量 |

loss：

`L_actor = -mean(min(ratio*A, clip(ratio,0.8,1.2)*A))`

`L_value = 0.5*mean((V-return)^2)`

`L = L_actor + 0.5*L_value - 0.01*mean(entropy)`。

rollout 以 no_grad 收集并保存行为策略 log-prob、value、动作、实际奖励、episode_start、terminated、最后状态、原始 uint8 图像。GAE 的递推不能越过 episode 终点；rollout 截止不是终止，需要 bootstrap。

### 循环状态与截断

- 片段内部保持顺序，不能把 individual frame 打乱；仅打乱片段顺序。
- 片段起点保存行为策略 hidden state。主默认加最多 **16 步无梯度 burn-in**：从较早保存状态重放片段前缀，以当前参数重建学习片段起始活动。episode 开始以内不足 16 步就用可用长度，新 episode 从零开始。
- burn-in 不计 PPO loss，也不跨时间反传；随后 32 步才有梯度。不是对整个 episode 做 BPTT。
- burn-in 降低但不能完全消除旧参数 hidden state 的误差；缓存最早状态仍可能来自行为策略。burn-in=0/16 是后续敏感性测试，不声称完全 on-policy 的内部状态重建。
- 每个 minibatch 使用当前参数重放；旧 log-prob 保持不变。参数尚未更新时，序列重放与 rollout log-prob 应一致；参数更新之后不要求相同。
- episode reset 时所有神经元状态清零，GAE trace 也断开。padding 与 burn-in 的 loss mask 严格为零。
- 每个游戏决策更新一次神经元活动；若以后内部更新 K 次，实际 BPTT 长度是 K×T，须重新测资源。

H=128 比 smoke 的 H=32 长，epochs=4 比 smoke 的 2 多，burn-in 也是新增。因此**正式默认配置尚未做性能验收**，不能直接套用 534 模拟器帧/秒。

## 6. 并行与性能验收

先跑固定 2,048 个游戏 transitions 的资源探针（不是学习预算主体），比较固定矩阵缓存前后，确认 W/Wᵀ 没有逐步重新生成。

为公平比较并行度，保持一次采样总量 512：测试 P/H=4/128、8/64、16/32；T=32、PPO epoch=4 不变，独立测训练 minibatch 的序列数 M=4/8/16。不要同时扩大总 rollout 来混淆吞吐改善。

选择端到端 decisions/s 最好、整卡显存至少保留约 1.5 GiB、无明显抖动的配置；性能差异 <5% 优先更少环境/更小 batch。各实验组使用同一最终选定 P/H/M/T，选择依据仅限性能，不看学习成绩。若资源或片段/burn-in组织限制使部分配置无法公平实现，记录未测并保留 4/128/4。

缓存图、增大批量和减少 smoke 的逐步 synchronize 可以加速，但要保留周期性有限性检查。资源验收与结果文件中必须注明同步/检查频率。gain 模式没用到边值梯度，所以不能宣称“本组有 Triton 5.5×加速”。

保留模拟帧、策略决策、有效训练 transitions、optimizer steps 四种计数；action repeat 固定为 4，不能通过增大 action repeat 冒充同控制精度下的加速。

## 7. 实验组与运行顺序

### A. 先做 GAIN 探索性实验

- seed=0，先跑 **200,000 次训练决策**；每 25,000 次做一次验证。
- 若未出现任何一圈成功，先停止并检查进度轨迹、动作可达性、输入到输出的活动与梯度。不能只凭零成功把结论归因于连接组无用。
- 若至少有一圈成功且无正确性问题，可按预先规则扩展到 **500,000 次总决策**。这是上限，不自动升级到千万步。
- 一圈门槛：验证 10 个回合中至少 5 个完成一圈，且至少在两个连续验证点满足。它决定是否有稳定进展，不替代五圈主指标。
- 此 seed 用于调试；不计入确认性结果。任何 reward、模型映射或超参数变更升版本并记录原因，不覆盖旧结果。

### B. 确认性基线矩阵

准备项通过、版本冻结后，每组训练 seeds={11,22,33}，每 seed **1,000,000 次训练决策**：

| ID | 网络与训练范围 | 目的 |
|---|---|---|
| GAIN | 真实图、固定 W、学习 gain+encoder+heads | 主实验 |
| FROZEN | 同真实图，W/gain 固定，学习 encoder+heads | 检查内部 gain 的增益 |
| GRU | 同 CNN，GRU hidden=256，actor/critic 从 GRU 状态读取 | 常规可学习策略基线 |
| REWIRE-GAIN | 度保持随机重连图，其他同 GAIN | 检查真实连接结构的作用 |

先运行 GAIN，再 FROZEN/GRU，最后重连组；无需同时并行多组训练争抢 GPU。总预算 12 个 run、**1,200 万次训练决策**，确认阶段是否全部执行由后续启动任务决定，本文件不自动启动。

GRU 是实际工作基线，不宣称神经元数/参数量/计算量严格匹配。报告每组 trainable parameters、总图存储、每决策 FLOP proxy（可选）、采样步数和 wall time。科学比较的主预算按 environment decisions 匹配，并补充 wall-clock 曲线。

### 随机图的约束

在固定节点集合上使用 directed double-edge swap：`u→v, x→y` 替换为 `u→y, x→v`，拒绝重复边及不允许的新自连接。保留每节点 in-degree/out-degree；不移动节点属性、输入掩码和输出节点。

为保持归一化输入强度分布，把原目标 v 的边权留给新 `x→v`，原目标 y 的边权留给新 `u→y`，从而保持每目标的输入权值多重集/行和；不声称同时保持每源输出权重和或类型间连接数。原自连接可单独固定并报告。

目标至少 5E 次成功交换，保存混合程度统计（边重合率、拒绝率、度与行和不变断言、输入到输出可达性变化）；5E 只是预设目标，不证明独立均匀采样。若计算成本过高，不得偷偷改成任意 Erdős–Rényi 图；另出预算并明确变更。

重连 graph seeds={101,102,103} 分别配对三个训练 seed。这只有三个“图+训练”联合重复，不能区分两类方差。若要更强拓扑结论，再扩展 graph seed×training seed 交叉设计。训练前检查可达性，不能依据游戏表现挑随机图。

### C. 后续可选组

FULL（全部现存边权学习）、T=16/64、不同输入映射、gain 类型共享、限制读出等均是后续独立实验。先完成 GAIN 的学习检查；FULL 若无 gain 就注明是“替代适应方式”，不要把它误说成 GAIN 严格嵌套组。

## 8. 评估、成功标准与 checkpoint 选择

### 验证集

- 固定初始存档，使用 stochastic policy，10 个固定且与训练独立的动作采样 seed={1000,...,1009}。评估不更新模型、optimizer 或归一化状态；使用独立 RNG，不消费训练动作 RNG。
- 探索阶段每 25k 决策一次；确认阶段每 100k 决策一次。评估决策/帧单独计数，不计入训练预算，但计入 wall time。
- 保留 last、best-validation、定期 checkpoint。best 首先按验证五圈成功率选，其次按成功回合平均时间，再按未成功回合合法进度；完全相同取较早者。
- 确认性主结果使用固定预算的 **final checkpoint**，best-validation 作为次要结果，避免只汇报峰值。

### 最终测试

对每个训练 seed 的 final checkpoint：

1. **名义任务主测试**：固定初始存档，stochastic policy，20 个预留、不同于验证的动作采样 seed={2000,...,2019}。
2. **确定性诊断**：固定原存档，argmax policy 运行一次。重复相同确定性 episode 不算独立样本。
3. **扰动稳健性**：20 个预留的轻微不同起始状态，argmax policy，生成 seed={3000,...,3019}。通过合法按键前缀生成，在可操控阶段产生小位置/方向/速度变化；生成后用画面、RAM 和存档哈希确认状态确有差异，均处于第一圈开头且可恢复。前缀、生成 seed 和存档哈希在测试前冻结，不参与调参。

扰动状态有自己的合法进度起点，计算“从该合法起点到五圈终点”的剩余任务；报告其定义，不能将不同起点的时间直接与原起点拼成一个完赛时间均值。主要计时指标仅用原起点主测试。

主指标：五圈成功率；补充成功条件下的完赛时间（中位数/IQR）、超时率、首次完成一圈的训练步数、合法最大进度、碰撞/离路/逆行事件。

“GAIN 学会名义任务”的预设判断：3 个 seed 中至少 2 个，在其 20 个主测试回合中成功 ≥16 次（80%）。这是工程门槛，不等于显著优于对照。扰动成功率单独报告，不因名义成功就称为泛化。

报告逐 seed 结果和三个 seed 均值/范围；episode 成功率可附 Wilson 区间，但不能把共享同一模型的 60 个 episode 当成 60 个独立训练实验。比较 GAIN/FROZEN/REWIRE 以配对 seed 差异为主，3 seeds 只支持初步结论，必要时扩大到 5–10 seeds。

训练中达到成功门槛也不提前结束确认性预算，避免不同组样本数不一致。探索性早停与确认性固定预算分开。

## 9. 开跑前必须通过的验收

| 验收项 | 通过标准 |
|---|---|
| 完赛与任务可达性 | 至少一条完整合法五圈轨迹；画面/圈数/checkpoint一致；成功只触发一次；限定动作集合可完成 |
| 进度防刷 | 起跑不奖励、前进/后退/回绕/往返同线/落水恢复/重置轨迹均有明确期望结果 |
| reset 与 recurrent mask | 强制制造终止；下一 episode hidden 清零；不把终止前奖励/GAE串到下一局 |
| 超时与 bootstrap | 成功/失败/任务超时 bootstrap=0；仅 rollout 截止正确 bootstrap；禁止用 auto-reset 画面估终态价值 |
| gain 梯度 | W requires_grad=False 且哈希不变；gain、CNN、heads 都有有限非零梯度，a 更新后 g 仍在范围内 |
| 序列重放 | 未更新参数时重放 log-prob 最大误差 <2e-5；padding/burn-in没有loss |
| 图索引与资源 | 真实方向、节点数、边数、input/readout mask无误；目标 H/T/M/epochs配置通过短测 |
| 日志与恢复 | checkpoint 恢复后参数、optimizer、计数器、RNG一致；环境/hidden无法精确恢复时标记新episode边界，不冒充无缝续跑 |

现有 smoke 只完成了其中部分。不得在缺乏任务判定证据时，用上游 reward/done 自动替代。

## 10. 预算、停止条件与记录

### 计算预算

探索阶段：200k，条件通过再到500k。确认阶段每 run 1M，12 runs 共12M。完整 episode 评估有额外成本；若全超时，10 个验证回合最多约45k决策，确认阶段10次验证最多额外450k决策/run，不能忽略。

以旧 smoke 133.5 decisions/s 仅作算术参考：1M训练决策约2.1小时；**正式方案的更多 epoch/burn-in、gain缓存、并行度和评估都会改变时间，这不是实际工期承诺**。正式短测后按真实吞吐估算每阶段 wall-clock，报告训练和评估各自时长。禁止从228ms合成更新直接推算学习时间。

### 自动停止与故障处理

- 达到指定阶段步数预算，停止采样并保存结果。
- NaN/Inf、错位序列、异常 progress、W意外更新、状态解析失败：立即停止并保存诊断；不自动继续训练掩盖错误。
- OOM：记录当前配置与peak；关闭run，不偷偷减batch改变既定实验。修订配置需新run ID/版本，并对相关组一致应用。
- 连续出现重复终止或游戏模式不符：停下检查存档/包装器。
- reward 上升但合法进度/完赛不升：先检查奖励漏洞，不能宣称任务改善。
- 探索失败必须保留结果；不能不断换seed只保留成功者。

### 每个 run 保存

`config.json`、代码/依赖版本、数据/ROM/state/映射哈希、所有seed、参数量、`metrics.jsonl`、checkpoint、评价逐回合结果和代表性视频。至少记录：

- 环境决策/模拟帧/optimizer steps/评估步数，所有终止原因。
- raw reward、有效进度、圈数、成功率、时间、动作频率、entropy、KL、clip fraction。
- policy/value loss、explained variance、gradient norm、gain范围/分位数/饱和比例、隐藏活动范数/饱和比例。
- rollout/forward/backward/optimizer时间，活跃/保留显存、整卡显存、CPU/RAM、每秒决策。

冻结实验中定期验证 W 不变，最终记录完整哈希；checkpoint 可只保存可训练参数并引用固定图哈希，但resume必须校验图一致。

## 11. 执行顺序

1. 实现正式任务 wrapper，完成完赛、进度、reset/GAE边界与动作可达性验收。
2. 实现 GAIN 模式、固定矩阵缓存、正式 recurrent PPO 序列/burn-in与日志；通过梯度与恢复检查。
3. 短时比较 P/H/M 的吞吐并冻结配置，输出正式预算估计。
4. 运行 seed=0 的 GAIN 探索阶段，按既定规则停止或扩展。
5. 版本冻结后运行三个seed的 GAIN、FROZEN、GRU、REWIRE-GAIN，完成独立测试。
6. 依据结果决定 FULL 或模型映射扩展，另立实验版本。

当前交付仅为完整方案。现有 rl_smoke.py 是有上限的验证脚本，不作为正式训练入口；没有开启任何新训练进程。
