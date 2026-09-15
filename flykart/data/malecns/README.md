# MaleCNS v1.0 原始文件

官方入口：https://www.janelia.org/project-team/flyem/male-cns-connectome

官方下载页：https://janelia-flyem.github.io/male-cns/download/

下载日期：2026-09-12。仅下载以下三个文件，均来自官方 GCS 前缀：
`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

| 文件 | 字节数 | MD5（与 GCS x-goog-hash 一致） |
|---|---:|---|
| connectome-weights-male-cns-v1.0-minconf-0.5.feather | 1051241946 | f30e9dcca25cfd021bf1e7b3d975599e |
| body-annotations-male-cns-v1.0-minconf-0.5.feather | 14483314 | 50a7718770c57220f160ba4f431ab89e |
| body-neurotransmitters-male-cns-v1.0.feather | 43282834 | 3d842b12fe5c49eefade528d7dd24a1f |

官方说明 weights 是包含所有有连接 segment 的完整 segment-to-segment 图，不能把所有 body ID 都认定为完整神经元。
annotations 是 curated neuron annotations；neurotransmitters 是 neuron/body 聚合预测。
未下载 EM、meshes、skeletons、synapse 坐标、partner 表或 body-stats。
数据许可 CC-BY 4.0，使用和发布时应引用官方数据与论文。

复查（从 flykart 目录）：

```bash
.venv/bin/python tests/inspect_malecns.py
.venv/bin/python tests/check_malecns.py
```

第一个脚本完整读取并验证 Arrow 表，打印 shape、columns、dtypes、前三行和空值统计。
第二个脚本按已观察的实际 schema 做基本一致性检查，输出 tests/malecns_summary.json。
