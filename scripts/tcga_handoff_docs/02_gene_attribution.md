# 第二步：主导轴特异性基因风险归因

## 本步骤回答什么问题
第一步确定了患者的功能风险状态；本步骤进一步解释：哪些基因通过该状态的主导功能轴贡献了模型风险？这是gene → axis → risk的模型归因，不是表达差异分析。

## 输入与计算过程
输入为对应患者的原始TPM、模型要求的基因顺序、表达到模型图像的映射资源、冻结的ME预测模型，以及第一步已有的主导轴Shapley贡献。上游预处理必须与模型一致，不能直接把任意标准化表达当作TPM。具体历史模型及基线记于`methods.json`，ME检查点为`training_checkpoints/ME/ckpt-20`。

先对“输入基因到指定ME输出”计算积分梯度（IG）。本批次记录为32个积分步；基线从相应fold/癌种训练样本均值图像出发，优化至历史OOF训练均值ME参考。它不是每个外部队列自行重置的中位数，也不是默认全零表达。对于患者i、基因g、主导轴a，最终分配为：

`gene_risk[i,g] = axis_shapley[i,a] × gene_IG[i,g] / sum_g(gene_IG[i,g])`

稳定分配时，各基因带符号贡献之和应接近该轴的Shapley值。这是沿主导轴进行的风险分配，不是直接对总风险输出做一次端到端IG；不要把二者称为完全相同的量。

## 质控与数值解释
实现检查IG总和与“样本ME减基线ME”的完整性误差，并检查分母是否过小。历史脚本默认最小绝对IG分母为1e-6、最大相对误差为0.20；重跑可能通过命令行改变默认值，实际结果应以QC表和运行记录为准。失败分配保留NaN，不得当作无风险贡献的零。

IG及风险归因不是天然0–1的概率。归因量较小不等于基因无用；方向依赖模型和参考状态。负向风险分配不表示基因低表达，也不等价于基因具有实验上已证实的抑癌作用。

## 文件如何读取
`dominant_axis_gene_logrisk_contribution.npy`为float32的患者×基因矩阵，对应9,579个样本及19,542个基因。它仅包含每位患者所属cluster的主导轴归因，**不包含每个患者的全部八轴gene→axis通道**。不同患者的主导轴可能不同。

`sample_order.tsv`给出矩阵行序，`gene_order.tsv`给出Ensembl列序。`samples.tsv`记录fold、癌种、cluster、轴方向、风险及用于关联的患者ID；`all_cluster_attribution_qc.tsv.gz`用于定位分配稳定性和完整性。不能依据文件名排序来推断矩阵顺序；其他表一律按sample_id连接。

```python
import numpy as np
import pandas as pd
from pathlib import Path
root = Path('02_gene_attribution')
x = np.load(root / 'dominant_axis_gene_logrisk_contribution.npy', mmap_mode='r')
samples = pd.read_csv(root / 'sample_order.tsv', sep='\t')
genes = pd.read_csv(root / 'gene_order.tsv', sep='\t')
assert x.shape == (len(samples), len(genes))
valid_rows = np.isfinite(x).all(axis=1)
```

## 与图片和下一步的关系
本目录没有独立主图；归因矩阵进入第三步NMF，再通过Figure 5B展示反复出现的程序。不能从单个大归因值直接给某个程序命名。原始TPM、全部训练权重和中间图像不包含在本步骤中，需要与单独的数据/模型发布配套。
