# Super Mario Kart integration（使用用户本地 ROM）

来源：https://github.com/esteveste/gym-SuperMarioKart-Snes

固定提交：`f84a1a9999f779f3beed6d05b895ee64664100d1`；许可证见 `LICENSE.upstream`。
仅下载 `data.json`、`metadata.json`、`scenario.json`、`script.lua`、`rom.sha`、
`Notes.txt` 和 `MarioCircuit_M.state`。没有克隆整个仓库，也没有下载其中的 ROM、训练代码或训练依赖。

上游 README 说明 Mario 存档采用 Time Trial。这里只准备 Mario Circuit 一个存档。
JSON 已解析，Stable-Retro `GameData.load` 返回 True，存档 gzip 校验和解压成功。
后续用户提供 ROM 后，已通过实际 Time Trial 存档加载、5,000 步 headless 操作和 6 次 reset。详情见 ../notes.md。

用户提供的 SNES Super Mario Kart ROM 已匹配此 integration 的 `rom.sha`：
`47e103d8398cf5b7cbb42b95df3a3c270691163b`（SHA-1）。
`SuperMarioKart-Snes/rom.sfc` 已符号链接到 `/home/giaok/Downloads/Super Mario Kart/Super Mario Kart (USA).sfc`，原文件未改动。移动原文件会使链接失效。从 flykart 目录运行：

```bash
.venv/bin/python tests/check_retro.py --mario
```

该命令只执行随机按键 smoke check。另可运行 `.venv/bin/python tests/probe_mario.py` 保存固定按键测试的 RAM 与截图。已查看 Time Trial 画面，验证速度、位置、checkpoint、路面和逆行变化；未完成整圈/整场比赛，finish 实际触发仍待验证。

可用字段（以下保留映射说明；实测结果以 ../notes.md 第 4 节为准）：

| 信息 | 映射或上游解释 |
|---|---|
| 画面 | Stable-Retro RGB uint8 图像；实测 (224,256,3) |
| 按键 | SNES B/Y/SELECT/START/UP/DOWN/LEFT/RIGHT/A/X/L/R，默认 MultiBinary(12) |
| lap | `lap`，上游减 128；完成五圈用 `lap - 128 >= 5` |
| progress | `current_checkpoint`、`lapsize`、`totalCheckpoints`；上游组合 checkpoint 与 lap |
| speed | `kart1_speed`，原始 RAM 整数，尚未确认物理单位 |
| surface / off-road | `surface`，Notes.txt 有路面代码；需要实机确认 |
| backward | `isTurnedAround`；Notes.txt 用 0x10 位掩码，script.lua 用等于 0x10，二者并非完全等价 |
| collision | 没有独立 collision 事件字段；上游用 surface=128 判断墙，不能代表所有碰撞 |
| finish | 没有独立 finish 字段；上游按 lap 和 `getGameMode` 判断 |

还包含 course、位置、方向、计时等变量。`test4`、`currTest` 等是上游实验字段，未验证。
`scenario.json` 自带上游 reward / done Lua 引用，原样保留；本阶段未设计、修改或优化 reward。
注意上游 `isDoneTrain` 也可因撞墙、落水等终止，不能把 terminated 直接解释为完赛。
