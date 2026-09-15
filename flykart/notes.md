# 环境与数据可行性检查（2026-09-12）

最新执行记录：[gain_experiment_start.md](gain_experiment_start.md)。用户授权后已启动GAIN/seed=0的200k决策探索实验，实时进度在runs/gain_seed0_v1/status.json；先前“未启动训练”的描述对应早期阶段。

后续 PPO＋截断 BPTT 完整基线设计已写入 [baseline_experiment_plan.md](baseline_experiment_plan.md)：首组固定 W、训练每神经元 gain 和输入输出头。本次设计未启动训练。

最新修复：20 万步基线诊断见 [gain_seed0_diagnosis.md](gain_seed0_diagnosis.md)；状态一致性修复、短任务配置、4096 步验证和独立对照见 [repair_experiment_notes.md](repair_experiment_notes.md)。短目标通过 8/10，不等于学会完整赛道；未继续长训练。

后续同配置对照已完成：[matched_gain_and_vision_results.md](matched_gain_and_vision_results.md)。训练gain和冻结gain均7/10，固定首帧也均7/10；当前前3 checkpoint任务不能证明视觉闭环驾驶，需先升级任务验收再扩大训练。

多起点真实弯道任务已实现并完成可行性检查：[bend_task_validation.md](bend_task_validation.md)。较强扰动版本8个候选起点均由三动作参考控制器通过；旧模型正常/固定首帧均11/16，仍未证明视觉反馈收益。本轮只验证任务，没有新训练。

最新弯道训练已完成32768决策：[bend_training_32k.md](bend_training_32k.md)。验证成绩1/9→6/9→2/9→2/9，正常与固定首帧始终相同。保留16384步最佳checkpoint；尚未发现视觉反馈收益，训练已停止。

本阶段只做环境安装、随机按键 smoke check 和原始表检查。没有训练模型，没有实现 PPO、BPTT、SNN、LIF、surrogate gradient，没有优化 connectome、设计 reward 或构建 PyTorch tensor。

## 1. Linux / GPU 环境

Ubuntu 26.04.1 LTS，内核 7.0.0-31-generic，x86_64；Ryzen 9 7900X（12 核 / 24 线程），RAM 29 GiB，检查开始时可用约 16 GiB；项目磁盘可用约 647 GiB。RTX 5070，显存 12,227 MiB，NVIDIA driver 595.84。

`nvidia-smi` 显示驱动支持 CUDA 13.2，`nvcc` 为 CUDA Toolkit 13.3。CUDA Driver API 初始化、设备枚举、上下文创建、显存分配/释放均通过；未测试 CUDA kernel 编译或 PyTorch。未来编译 CUDA 代码时需确认 toolkit / driver 版本配合，当前 CPU 模拟和数据检查没有阻碍。

系统 Python 3.14.4；项目 `flykart/.venv` 使用 Python 3.12.14。已安装 requirements.txt 中六个基础库及必要依赖，`uv pip check` 通过。未安装任何 RL / 深度学习框架；原项目 main.py 和 IDE 文件未改动。环境约 672 MiB，三份数据合计约 1.03 GiB。

## 2. Stable-Retro

stable-retro 1.0.1（推荐 `import stable_retro as retro`）正常。使用包自带 Airstriker-Genesis-v0，清除 DISPLAY / WAYLAND_DISPLAY，以 rgb_array 模式随机操作 5,000 步，6 次 reset，无异常；所有 reset 恢复相同初始画面。观测 `(224, 320, 3)`、uint8 RGB，action space `MultiBinary(12)`。循环约 1.68 秒，仅为 smoke check，不作精确 benchmark。`env.render()` 返回数组正常。

复查：在 flykart 目录运行 `.venv/bin/python tests/check_retro.py`，结果见 `tests/retro_smoke.json`。

## 3. Super Mario Kart / SNES / Time Trial

Mario Kart integration 和用户提供的 ROM 已成功运行，环境 READY（2026-09-12 更新）。

复用 [esteveste/gym-SuperMarioKart-Snes](https://github.com/esteveste/gym-SuperMarioKart-Snes)，提交 `f84a1a9999f779f3beed6d05b895ee64664100d1`。目录为 `mario_kart/SuperMarioKart-Snes/`，有 `MarioCircuit_M.state`、`data.json`、`scenario.json`、`script.lua`、`metadata.json`、`rom.sha`、`Notes.txt`；许可证已保存。

上游说明该存档采用 Mario / Time Trial。已验证注册为 custom integration、JSON 解析、GameData.load=True；存档可完整 gzip 解压（431,641 字节，SNES9x snapshot v0009）。SNES9x core 已随 stable-retro 安装。

用户随后提供 `/home/giaok/Downloads/Super Mario Kart/Super Mario Kart (USA).sfc`，524,288 字节，SHA-1 `47e103d8398cf5b7cbb42b95df3a3c270691163b` 与 integration 完全一致。已在 integration 的 `rom.sfc` 建立指向原文件的符号链接，未修改原 ROM、未下载 ROM。移动 Downloads 中原文件会使链接失效。

实际 reset 成功加载 Mario Circuit / Mario 的 Time Trial 起跑存档；已查看单人计时赛画面（计时器、单车赛道地图），与上游 Time Trial 说明一致，RAM GameMode=2、getGameMode=28。这里验证的是从现有存档进入比赛，未测试从开机菜单逐级导航。

无 DISPLAY / WAYLAND_DISPLAY 下随机按键运行 5,000 步，6 次 reset 均恢复完全相同画面，耗时约 3.71 秒，无异常。观测为 `(224,256,3)` uint8 RGB，`env.render()` 返回数组正常。额外对照“不按键 / 持续 B / 持续 B+RIGHT”确认加速与转向有效：不按键速度为 0；B 使速度升至 708、位置和 checkpoint 变化；B+RIGHT 使轨迹改变并触发逆行提示。持续 B 第 400 步撞墙，被上游 scenario 正常终止；再次 reset 成功。没有完赛，不声称已验证 finish 的实际触发。

复查：`.venv/bin/python tests/check_retro.py --mario` 和 `.venv/bin/python tests/probe_mario.py`。证据见 `tests/mario_smoke.json`、`tests/mario_probe.json`、`tests/mario_frames/`。reset 返回的 info 是空字典，RAM 字段在 step 返回的 info 中可读。

## 4. Mario Kart observation / game-state

图像接口是 RGB uint8；SNES 按键为 B/Y/SELECT/START/UP/DOWN/LEFT/RIGHT/A/X/L/R，默认 `MultiBinary(12)`。实际状态验证如下：

| 状态 | 字段与限制 |
|---|---|
| lap | `lap` 实测 127→128（起跑前→越过起跑线）；上游减 128；未完成整圈 |
| checkpoint / progress | `current_checkpoint` 实测 29→0→2，`lapsize`=30；连续 progress 可由上游公式解释 |
| speed | `kart1_speed` 实测 0–708，随加速变化；物理单位未确认 |
| surface / off-road | `surface` 实测道路 64、泥地 84、墙 128，与截图一致；其他地形未逐一验证 |
| backward | `isTurnedAround` 实测 0/16，与逆行提示画面一致；其他位组合尚未验证 |
| collision | 撞墙画面对应 surface=128，并触发上游 done；仍无通用碰撞事件字段 |
| finish | 无独立 finish 字段；上游使用 `lap-128 >= 5`，未实测完赛，不能把 done 直接当 finish |

另有 course、位置、方向、计时字段。上游 scenario / Lua 原样保留，其 reward / done 逻辑未改写；没有开展 reward 设计。详情见 `mario_kart/README.md`。

## 5. MaleCNS 下载与读取

三个指定 Feather 已从 [Janelia 官方下载页](https://janelia-flyem.github.io/male-cns/download/) 对应的 `storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/` 下载至 `flykart/data/malecns/`。本地文件长度、MD5 与官方 GCS 一致；pyarrow 完整验证和 pandas 转换均成功，没有空表或损坏文件。下载源与校验值见 `data/malecns/README.md`。没有下载其他大型数据。

## 6. 实际 schema

`tests/inspect_malecns.py` 已对每张表打印 shape、全部 columns、dtypes、前三行和空值数；先观察 schema，再进行列映射。

**connectome weights：151,856,684 行 × 3 列**

`body_pre`（source，int64）、`body_post`（target，int64）、`weight`（连接突触计数，int64）。前三条为 `(10352,10351,2591)`、`(13612,16076,2443)`、`(10663,10051,2248)`；三列无空值。

**annotations：211,577 行 × 36 列**

```text
assignedOlHex1, assignedOlHex2, bodyId, flywireType, group, instance,
somaSide, statusLabel, superclass, type, vfbId, hemibrainType, itoleeHl,
supertype, birthtime, mancBodyid, mancGroup, mancType, subclass, synonyms,
class, rootSide, somaNeuromere, trumanHl, dimorphism, matchingNotes,
entryNerve, mancSerial, mcnsSerial, serialMotif, fruDsx, exitNerve,
receptorType, somaLocation, tosomaLocation, status
```

ID 为 `bodyId`（int64）；type/instance/supertype 和跨数据集 type 字段描述类型；class/subclass/superclass 描述类别；somaSide/rootSide 描述侧别；somaNeuromere、entryNerve/exitNerve、somaLocation 等提供区域/解剖线索，**没有通用 region 列或完整连接 ROI 表**。status/statusLabel 描述 tracing 状态。`mancBodyid` 是另一数据集映射，不能替代本图 ID。

dtypes：assignedOlHex1/assignedOlHex2/group/mancBodyid/mancGroup/mancSerial/mcnsSerial 为 float64；statusLabel 为 category；somaLocation/tosomaLocation 为 object；bodyId 为 int64；其余为 str。可选元数据大量缺失，bodyId 无空值。

**neurotransmitters：1,835,518 行 × 10 列**

```text
body, cell_type, total_nt_predictions, predicted_nt_confidence,
predicted_nt, ground_truth, celltype_total_nt_predictions,
celltype_predicted_nt, celltype_predicted_nt_confidence, consensus_nt
```

`body` 是本图 neuron/body ID（int64）；两列 total_nt_predictions 为 int32；两列 confidence 为 float64；其他为 str。predicted_nt 是 body 聚合预测，celltype_predicted_nt 是细胞类型级预测，consensus_nt 是共识结果，ground_truth 为已有真值信息，不能把这些字段都当实测标签。body 无空值。

## 7–9. 数量、一致性与下一步

完整统计见 `tests/malecns_summary.json`。

| 指标 | 实测结果 |
|---|---:|
| 原始完整图 edge 数 | 151,856,684 |
| 原始图 unique body / segment ID | 88,384,522（**不是完整神经元数**） |
| source / target unique ID | 1,834,661 / 87,576,984 |
| 注释表 unique ID | 211,577；其中 status=Traced 为 165,122 |
| 有注释且出现在图中的 ID | 191,696，占完整图 ID 的 0.2169% |
| 两端都有注释的 edge 数 | 26,028,386（仅统计，未生成子图） |
| 图 ID 有递质记录 | 1,835,518，占 2.0767% |
| 图中已注释 ID 有递质记录 | 187,016 / 191,696，即 97.5586% |
| weight 范围 / 中位数 / P90 / P99 | 1–2,591 / 1 / 3 / 15 |
| 非正 weight / 空 ID / 重复注释 ID / 重复递质 ID | 均为 0 |
| 自连接边 | 123；需后续按生物学规则判断，不自动删除 |

未发现文件损坏或明显异常的非正边权。94,185,919 条边权为 1，全部边权合计 311,833,243。低全图注释覆盖率与官方“所有 segment 的完整图”定义一致，不是 join 列错误。注释表还包含 Glia（11,864）、Orphan、Unimportant 等，211,577 也不能直接当作神经元数量；本阶段可用的近似神经元规模是 **约 16.5 万条 Traced 记录**。19,881 个注释 ID 不在此连接图中，后续需结合状态筛选处理。

递质“有记录”不代表预测明确：predicted_nt 中 1,684,226 条为 unclear，consensus_nt 中 1,671,117 条为 unclear。以注释表的 165,122 条 Traced 为分母，164,620 条有递质记录（约 99.70%），161,520 条有非 unclear 的 consensus（约 97.82%）。body-level confidence 范围 0.186858–0.975368，有 857 个空值，非空值无超出 [0,1] 的情况；celltype confidence 有 1,671,073 个空值。多数可选注释缺失（type 有 164,506 条非空），这会影响未来筛选/解释，但不妨碍当前读取验证。

资源方面，完整图检查可完成，但全量 pandas + ID 集合运算曾占约 13 GiB RSS，并使机器使用约 2.8 GiB swap；后续宜分批读取或先定义有注释的神经元范围，不宜盲目全量建模。

下一步仅建议：另行确认 Traced / 类型 / 递质置信度等数据筛选规则；如果后续需要可靠的完赛标志，再完成一次整场 Time Trial 对照验证。**当前基础环境验证已完成；本次没有应用筛选、构建网络或开始训练，到此停止。**

```text
MARIO_KART_ENV = READY
MALECNS_DATA = READY
```

## 后续计算可行性验证（2026-09-12）

最新：用户授权的真实训练 smoke test 已通过，使用 165,122 个 Traced 节点、25,563,197 条实际连接与真实 Mario Kart ROM，完成 512 次决策、8 次 PPO 更新。峰值张量显存约 1.55 GiB，端到端约 534 模拟器帧/秒。详情及限制见 [rl_smoke_results.md](rl_smoke_results.md)。这次是限定时长训练验证，未进入长期学习实验。

更新：三种算子路线及 Triton 融合测试已完成；避免 N² 中间量后，165,122 节点的合成全边权更新已通过。实测与限制见 [sparse_operator_results.md](sparse_operator_results.md)，不等于真实 MaleCNS RL 已验证。

用户随后授权小规模计算/显存验证，已在独立 `.venv-bench` 环境对合成稀疏循环网络做 forward、时间展开反向传播和少量 Adam 更新，没有进行游戏 RL 训练，也没有转换或修改 MaleCNS 数据。19 个配置中 17 个通过、2 个达到受控显存上限。实测结果、限制和下一阶段计划见 [compute_feasibility_and_plan.md](compute_feasibility_and_plan.md)。前文“未实现训练/时间反传”描述的是初始环境准备阶段。

## 最新视觉诊断（2026-09-13）

用户要求尝试逐层视觉敏感性与扩展任务/开环对照，均已完成。正常/固定首帧/途中冻结/纯时间表各12/30通过；纯时间表28/30局动作序列与正常完全相同。三checkpoint逐层诊断显示训练后场景变化相对信号弱，读出端更弱，无encoder饱和。6个新起点已通过三动作参考控制器，仍未证明任务必须依赖持续视觉。详情见 [bend_feedback_diagnosis.md](bend_feedback_diagnosis.md)。模型参数和checkpoint未改，未启动新训练。

## 架构与RL机制审计（2026-09-13）

用户要求回到架构和RL机制。完成结构可达性、收缩界、训练日志检查，以及既有最佳/最终checkpoint各3个无更新PPO梯度批次（共768决策）。全部读出视觉最短路径为1–3条边；没有同决策视觉输入；当前隐藏状态差异收缩上界约0.9003。6个批次中5个actor/critic编码梯度冲突，但强度不一，尚非因果结论。建议转为“现有图/标准GRU × 动作监督/PPO”对照，先检验明确监督下能否学会状态相关动作。详情见 [architecture_rl_review.md](architecture_rl_review.md)。未启动新训练、未修改模型或checkpoint。

## 监督学习架构对照完成（2026-09-13）

用户授权执行。相同CNN初值、示范、批次顺序及256更新，固定图与CNN+GRU各3seed；测试17起点，完成1224局反馈评估。GRU正常/固定首帧/时间表greedy通过44/51、9/51、3/51；固定图7/51、12/51、19/51。测试平衡准确率GRU63.7%、图36.2%。持续视觉控制在标准架构下可学，当前图架构/优化组合是优先瓶颈，不能单独归咎PPO；也未证明任意connectome不可学习。全部模型、数据、配置保存在runs/supervised_architecture_v1，详情见 [supervised_architecture_results.md](supervised_architecture_results.md)。PPO一列尚未执行；旧模型checkpoint完整性已验证。

## 可训练视觉接口对照完成（2026-09-14）

用户授权继续，完成11组按视觉superclass×somaSide的identity初始投影，共新增45056参数；固定W/动力学/读出/监督预算，3seed×256更新及612局闭环。新接口测试平衡准确率36.4%，旧接口36.2%；正常/固定首帧/时间表greedy通过12/51、16/51、21/51，未出现视觉收益。共享初值、样本呈现数、初始前向等价、梯度及checkpoint完整性检查通过。详情见 [visual_injection_results.md](visual_injection_results.md)，数据在runs/visual_injection_v1。建议下一项独立检查1/2/4内部更新，尚未执行。

## 神经内部更新次数对照完成（2026-09-14）

用户授权继续。同游戏repeat8，试点k=2/4各seed0；按预设验证规则选择k2并复验seed1/2。共4组新监督训练、816局闭环，全部检查通过。k2测试平衡准确率39.8%（原36.2%），greedy正常/固定首帧/时间表9/51、4/51、11/51；未获得稳定的视觉驾驶。k4单seed测试34.6%，greedy3/17、2/17、0/17。当前画面到当前动作梯度从0变为非零，但任务仍未解决。详见 [microsteps_results.md](microsteps_results.md)，输出runs/neural_microsteps_v1。建议下一步做分层表示可解码性探针，尚未执行；未启动PPO。

## 冻结分层动作探针完成（2026-09-14）

用户授权继续，完成既有 k1/k2/GRU seed0 冻结骨干的 24 个线性/MLP 探针和 2 个纯时间探针；采用训练集标准化和原轨迹划分。k2 CNN/MLP 测试平衡准确率 73.5%（固定画面 41.1%），motor256/MLP 52.8%，中间节点/MLP 48.7%。按验证集选择 k2 CNN/MLP 与 GRU 隐状态/MLP，共完成 408 局闭环及 2 局顺序回归检查。k2 CNN 探针 greedy 正常/固定/时间通过 15/17、4/17、6/17，sampled 为 40/51、16/51、7/51；GRU greedy 为 16/17、8/17、6/17。CNN 探针绕过图网络，证明现有视觉编码可支持当前任务，不代表图网络已经成功；单 seed、复用测试起点，仅作诊断。所有骨干参数及旧 checkpoint 不变，特征/部署一致性与梯度隔离检查通过。详情见 [layer_probe_results.md](layer_probe_results.md)，数据 runs/layer_probes_v1。下一步优先比较保留图路径的标准化读出，并以 CNN 直连动作头为正对照；尚未执行新架构或 PPO 训练。

## 冻结 motor 读出尺度对照完成（2026-09-14）

继续完成 k2 seed0 原始/标准化完整 motor 线性读出的匹配优化对照，复用上一轮标准化模型并追加 motor256/MLP 闭环，共 612 局及 3 局顺序回归。标准化测试 BA 从 45.6% 升到 48.4%，greedy 正常通过却从 5/17 变为 3/17；两者固定为 0/17、2/17，时间均 6/17。motor256/MLP greedy 正常/固定 9/17、8/17，sampled 7/51、20/51，不构成可靠视觉驾驶。旧权重、骨干梯度隔离、特征部署一致性、批量顺序一致性及共同时间序列检查通过。详见 [readout_conditioning_results.md](readout_conditioning_results.md)，输出 runs/readout_conditioning_v1。仅事后调整读出不足；下一项建议是匹配的端到端原始/标准化读出监督对照，以检查反传条件能否改善视觉与增益学习，尚未执行。

## 三项架构假设试点完成（2026-09-14）

用户授权执行空间映射、符号与衰减验证。数据含23720个视叶hex坐标节点（视觉节点23%）；保守高置信GABA负号15914节点。完成15组不同历史/相同后续输入洗出测量，k2全状态差异最大值16决策后中位剩1.73%、32后0.044%；减慢更新或增益×1.2确能延缓衰减，但不等于驾驶改善。固定同一个k2 seed0 CNN，7组各128监督更新：baseline、spatial、spatial_shuffle、gaba、gaba_shuffle、slower、gain1.2。空间/打乱验证BA均44.4%，GABA41.5%/打乱42.1%，slower44.1%；gain1.2验证49.8%过预定门槛，但新测试42.7%低于baseline43.2%。新增18起点oracle可行16个，完全相同帧与旧示范交集0，仍同赛道。完成baseline/gain1.2/旧CNN正对照各192局，共576局及3局顺序回归。greedy正常/固定/时间分别4/16、2/16、6/16；1/16、1/16、6/16；15/16、1/16、6/16。CNN sampled正常38/48、固定2/48，视觉正结果在新增起点重现。所有原文件与CNN冻结检查、符号算子及空间干预检查通过。详见 [architecture_mechanisms_results.md](architecture_mechanisms_results.md)，runs/architecture_mechanisms_v1。三种具体干预均未修复图网络，单seed短预算且从已训练全正模型出发，不能排除其完整版本。下一项建议固定CNN任务特征作为重建监督来检验视觉到motor的可训练传输，尚未执行。

## 视觉表示重建与梯度诊断完成（2026-09-14）

用户授权检验大图训练/梯度是否有效。固定既有k2 seed0 CNN，3条训练轨迹前24帧（72目标），6条分开验证轨迹前21帧（126目标），重建64维标准化CNN特征；4组各400全批量监督更新：decoder_only、gain、interface、interface_edges。最终训练/验证MSE依次0.290/0.264、0.213/0.214、0.023/0.125、0.024/0.130。接口为512参数、固定随机N×8基底残差；受限边6092条各±50%，边调整未进一步改善。接口固定首帧验证MSE1.414，显示利用持续视觉变化。闭式固定motor读出可近乎插值训练点，但弱正则过拟合；ridge0.1验证MSE0.102也是强参考，不能断言接口唯一有效。首步Adam过冲触发独立匹配学习率对照：lr.003/.0003/.00003最终训练MSE.243/.189/.381。接口和边权有限差分检查通过，参数更新、CNN冻结和源checkpoint完整性通过。重建特征接旧冻结CNN动作头，验证短窗口BA仅读出26.6%、gain36.3%、接口52.9%、接口加边49.5%；直接CNN76.2%。本轮未测驾驶。详见 [transport_reconstruction_results.md](transport_reconstruction_results.md)，runs/transport_reconstruction_v1。支持优化条件和可训练输入接口有改进空间，不支持“完全反传失败”或“图状态无信息”；建议完整数据的正则读出/接口预训练对照后接动作监督闭环，尚未执行。

## 完整轨迹抑制/重建对照完成（2026-09-14）

用户授权继续。相同冻结CNN、a=0/接口=0/重建读出=0，完整43训练轨迹2184决策，positive/GABA/shuffled各384更新，lr.0003；初值/样本呈现匹配。验证重建MSE .162/.178/.188，新测试MSE .164/.185/.189，动作BA45.1%/38.8%/40.6%。新增18起点oracle可行16；按预设规则选择positive/GABA闭环，另测旧CNN正对照，共576局＋3局顺序回归。greedy正常/固定/时间：positive5/16、4/16、3/16；GABA5/16、4/16、3/16；CNN13/16、3/16、3/16。sampled正常/固定：positive19/48、18/48；GABA8/48、8/48；CNN40/48、7/48。GABA降低状态饱和比例但没有改善视觉控制。本轮只特征重建后接冻结CNN动作头，未做动作监督适配；建议下一项同接口的特征重建对照动作蒸馏联合重建，尚未执行。所有原source/模型hash、CNN冻结、context/padding及批量顺序检查通过。详见 [inhibition_reconstruction_results.md](inhibition_reconstruction_results.md)，runs/inhibition_reconstruction_v1。

## 动作蒸馏联合重建对照完成（2026-09-14）

用户continue授权继续。从inhibition_reconstruction_v1/positive_best共同warm-start，两组各256更新、lr.0003，完整原轨迹，冻结CNN及教师动作头。reconstruction仅MSE；joint为MSE+KL(教师||学生)系数1，均按验证教师KL选checkpoint（均256）。验证BA59.1%/50.0%；新测试BA50.5%/43.7%，KL .500/.419。新增18起点oracle可行16，和前两轮诊断示范无相同帧。完成两图模型及CNN各192局，共576局+3顺序回归。greedy正常/固定/时间：重建12/16、6/16、7/16；联合11/16、0/16、7/16；CNN11/16、2/16、7/16。sampled正常/固定/时间：重建27/48、16/48、10/48；联合27/48、11/48、10/48；CNN34/48、15/48、10/48。出现保留图路径的单seed视觉闭环收益，但联合没有提高正常通过数，不能仅凭固定画面更差称优胜；未证明果蝇拓扑价值。梯度到接口/增益/读出、冻结teacher/CNN、初值/样本匹配、sourcehash及批量顺序回归全部通过。详见 [action_distillation_results.md](action_distillation_results.md)，runs/action_distillation_v1。下一步建议多seed固定方案复验，再比较真实/随机拓扑，尚未执行。

## 三采样种子共同起点复验完成（2026-09-14）

用户授权继续。复用seed0两模型，新增seed1/2两目标各256更新，保持同一CNN教师和warm-start，仅改变minibatch抽取序列（不等于独立预训练）；验证KL选择检查点。新增24候选起点oracle接受20，和历史成功示范无相同帧。六图策略+CNN共1680局及7顺序回归。greedy正常：重建14/20、6/20、10/20；联合15/20、13/20、11/20。联合固定2/20、5/20、4/20，时间均3/20；sampled正常37/60、26/60、28/60，均超过各自固定和时间对照。联合视觉收益跨三个采样种子出现，但seed2仅55%，低于预设60%门槛，未全部通过稳定性标准。重建也未过，且波动更大。CNN正常16/20、固定1/20、时间3/20，sampled46/60。不能外推独立预训练稳定性或果蝇拓扑价值。共同失败而CNN成功的联合起点为f455_left_p38、f505_left_p14。全部sourcehash、冻结教师/CNN、成对初值/样本匹配及批量顺序检查通过。详见 [distillation_replication_results.md](distillation_replication_results.md)，runs/distillation_replication_v1。下一步建议定位两个共同失败点的首次动作分歧与状态偏移，尚未执行。

## 2026-09-14 发布与共同失败轨迹诊断

首次发布到 origin/main，排除 ROM、游戏画面、模拟器状态、二进制数据/模型和环境；保留数值实验档案，第三方 integration 保留 MIT 许可。上游 Notes.txt 引用其他项目，暂不发布。详见根 README.md。

继续完成 42 局诊断，见 divergence_diagnosis_results.md。两个共同失败起点的六个 seed×起点组合均在第 0 或第 3 次决策分歧；短暂 CNN 接管 1/4 次分别挽救 1/6、2/6。延迟 8/32 次后持续 CNN 接管共仅 4/12 成功，说明直接 CNN 也缺乏可靠偏离恢复能力。12 条原始轨迹精确回归通过，所有 checkpoint 保持不变。

下一步先在训练起点验证恢复控制器和示范质量，再做匹配的恢复数据训练对照；不能把 CNN 对失败状态的预测无条件当作正确标签。暂不新增 E/I 或更复杂动力学。此处结论优先于此前未执行的下一步建议。

## 2026-09-15 训练起点恢复示范验证

完成 129 局自主基线 + 472 局接管，共 601 局，全部来自原 43 个合格训练起点。joint 三个采样种子自主成功 31/43、30/43、25/43。对其中 43 个失败组合，第 8 次决策接管：waypoint 43/43、CNN 29/43；第 32 次接管：waypoint 37/43、CNN 19/43。配对 86 组中仅 waypoint 成功 33 组，仅 CNN 成功 1 组。说明现有 waypoint 可以提供明显更完整的恢复示范，但属于特权状态控制器，不是视觉 fly 的自主成绩。

按预先规则得到 81 条合格恢复轨迹，精确去重后 74 条，覆盖 25 个起点、全部四个训练 prefix。监督动作 4,165 个（去重前），接管前失败动作均以 -100 屏蔽，真实完整观察前缀保留。覆盖门槛通过；前缀和接管画面、文件哈希、标签一致性检查全部通过。与原验证/测试及四轮归档新起点示范的精确画面重叠为 0。帧数据仅在本地，不发布。

下一轮增加损失匹配对照：A 旧数据 MSE+KL；B 旧数据混入动作 CE；C 与 B 相同损失和比例但混入恢复示范。B/C 区分数据收益和损失变化，全部保留冻结视觉/时间对照及三采样种子。尚未训练新模型，因此此轮不能宣称自主准确率提升。完整报告 recovery_demonstrations_results.md，数值档案 runs/recovery_demonstrations_v1/。
