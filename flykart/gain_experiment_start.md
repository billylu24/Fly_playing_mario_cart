# GAIN seed=0 第一阶段实验启动记录

日期：2026-09-12。用户授权后已启动后台实验，run 为 `runs/gain_seed0_v1`，预算 **200,000 次训练决策**。此处为启动记录，不是完赛结果；实时状态以 run 内 `status.json` 为准。

## 已通过的启动验收

- 只用允许的 B/Y/LEFT/RIGHT 组合、每动作保持4帧的特权路线控制器完成五圈，8462个模拟帧（含起跑），raw lap最终133，合法进度150段。控制器仅用于验证，不生成训练示范，不接入策略。
- 从 SNES WRAM 块 `0x7e0000` 读取赛道地图用于验收；不能用 `get_ram()` 拼接数组下标直接当真实地址。
- Y 刹车对照：相同起始速度592，20帧后不按键速度460、Y速度443。
- 用五圈逐帧轨迹回归 reward，起跑不奖励、回绕正确、成功奖励仅一次，总reward=7.8845，等于10−8462×0.00025。
- 合成往返/跳段测试：重复跨旧线不奖励；跳过未达到段不补记进度；必须从已验证前一段正向进入下一段。
- 真实超时后reset图像与初始图像逐像素一致；episode_start将全脑hidden清零；终止GAE不串入下一局。
- gain/encoder/heads梯度正常、gain有实际更新、W固定未变化；可训练参数357,378。
- checkpoint模型/优化器/RNG保存恢复验证通过，恢复训练从2048至2560决策完成。恢复时新建episode，明确记录边界，不假装恢复模拟器中途状态。
- 独立seed=1000评估跑满18000帧并正确超时，进度2/150；这是未训练好探针的结果，不是正式实验成绩。

详细证据：`tests/baseline_gates.json`、`tests/evaluation_gate.json`、`tests/track_validation/race_trace.json`、finish截图、`runs/gain_resume_check/status.json`。

## 正式实现与计划的落实

- `training/task.py`：独立reward/终止/超时wrapper；不采用上游撞墙done，原integration文件不变。
- `training/model.py`：固定CSR W/Wᵀ缓存、每节点有界gain、CNN、actor/critic；固定矩阵仅对输入反传，不求边值梯度。
- `training/train_gain.py`：PPO、GAE、T=32截断BPTT、最多16步burn-in、KL早停、有限性检查、日志、checkpoint、独立RNG验证。
- 新scenario为 `mario_kart/validation_scenario.json`（空reward/done定义），只读取已有RAM映射。
- 原始MaleCNS、ROM和上游scenario未改写。派生图、原始文件与代码哈希记录在run配置中。
- reward：每个新合法checkpoint +1/30；首次五圈完赛+5；每实际模拟帧−0.00025。300秒模拟时间是任务内终点，bootstrap=0。
- 每个rollout保存行为hidden并做未更新参数的精确重放检查；burn-in用当前参数重算，其旧hidden近似误差另记，不能将两种重放误差混为一谈。

## 并行选择

按相同总rollout=512、4个PPO epoch，测试了三个2048决策探针：

| 环境数 / 每环境rollout / 序列batch | 稳态decisions/s中位数 | 张量峰值MiB |
|---|---:|---:|
| 4 / 128 / 4 | 74.35 | 1218.5 |
| 8 / 64 / 8 | 142.88 | 1445.7 |
| 16 / 32 / 16 | 275.32 | 1907.8 |

第一项部分测量期间另跑了验证进程，可能有轻微GPU竞争，因此不将其作为精确加速倍数证据。选择16/32/16是当前候选中最快且有余量者，不声称完整9组合网格的全局最优；未测的交叉组合明确保留为后续性能工作。

增加序列batch会减少每个epoch的optimizer step数量：16环境配置每epoch一个minibatch，四个epoch每轮四次更新。样本重复次数仍为4，但优化轨迹不同；后续对照组必须采用同一配置。没有把这一改动当作纯算子加速。

性能探针只用于选配置；正式run从seed=0重新初始化，不继承探针权重或经验。已选配置：**16环境、H=32、T=32、M=16、burn-in≤16、PPO epochs=4**。

## 预算、输出和运行状态

- seed=0，严格200,000决策后保存停止；不会自动扩到500,000。
- 每25,000决策触发验证，实际在第一个满足阈值的rollout边界执行（例如25,088）。每次10个stochastic evaluation episodes，seed=1000…1009，评估步数不计训练预算。
- 验证可能较慢：独立单局超时测试约19秒，10局全超时约3分钟；此期间status为evaluating，训练计数暂不增长是正常现象。
- 按275 decisions/s仅训练计算约12分钟，另有8次验证、哈希和保存；粗估总时长约35–45分钟，实际受策略进度、CPU/GPU负载影响，不保证。
- 每约10,000决策及每次验证前保存last.pt；保存initial.pt、best_validation.pt；异常保存interrupted.pt并标为failed。训练日志、评估逐局结果和状态持续写入。
- 正式日志含loss、KL、entropy、动作频率、gain范围/分位数、hidden活动、有效进度、梯度范数、显存与吞吐。v1尚未自动导出评价视频或GPU功耗/CPU监控时间序列；checkpoint可另行回放，不将这些诊断遗漏视为已完成。

查看：

```bash
tail -f runs/gain_seed0_v1.log
cat runs/gain_seed0_v1/status.json
```

从flykart目录手动复现同等新实验（必须使用新run目录）：

```bash
.venv-bench/bin/python -m training.train_gain --run runs/gain_seed0_new \
  --budget 200000 --envs 16 --sequences 16 --seed 0 --eval-every 25000
```

可用 `--resume <checkpoint>` 恢复模型/optimizer/RNG/计数器到新的run目录，环境从新episode开始；用于可追踪的恢复段，不是无缝重放原模拟器。原run文件不覆盖。
