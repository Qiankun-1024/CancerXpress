# 第三步：风险状态内部的NMF分子程序（Figure 5B）

## 科学问题和输入
同一个功能轴主导的风险状态是否包含不同的基因贡献模式？每个Cxx独立拟合NMF，输入来自第二步该状态患者的主导轴基因风险归因，不是表达矩阵，也不是八轴混合后的总体基因重要性。

## 数据处理和模型
去除含非有限数值或总绝对贡献近零的患者。每位患者先用全部基因绝对贡献总和做L1归一化；再按cluster内平均绝对归一化贡献选择前1,000个基因。这1,000个候选基因是该cluster共同的NMF输入，不是每个program独立挑选1,000个。

每个基因拆成正、负两个非负通道：X+ = max(X,0)，X− = max(−X,0)。最终输入约为患者×2,000列，正通道在前、负通道在后。负通道的数值本身为非负数，表示负向归因的强度。用于拟合的矩阵乘以1,000作数值缩放。

NMF使用nndsvda初始化、coordinate descent求解和Frobenius重建损失；该函数没有显式传入额外稀疏惩罚。每个候选rank进行多次初始化，保留重建误差最小的结果。H每个因子按所有通道的总loading归一化，并相应调整W以保持重建。随后W按患者行归一化，得到usage_fraction；同一患者程序usage之和约为1。这是程序相对使用量，不是患病概率或各程序的生存风险。

## 程序数和稳定性如何得到
各cluster的候选rank、重建解释比例和bootstrap指标在`rank_diagnostics.tsv`。对bootstrap程序与参考H通过余弦相似度和Hungarian匹配，比较loading及核心基因重叠。此处bootstrap会重新选择候选基因，再映射到参考特征空间，和第六步固定候选基因的C11分析不同。

rank选择先筛稳定性门槛；若没有rank通过则退回所有候选rank。在候选中按重建曲线knee分数、loading余弦、较小rank依次排序选择。因此不是人为统一规定所有cluster四个program。图示稳定program标记采用余弦≥0.80且core Jaccard≥0.50。配置数值和每次结果仍应以对应诊断文件为准。

## Figure 5B及附图怎么读
主图分别展示C09/MEpink+、C11/MEred+、C03/MEblue−；完整附图包含其他cluster。每列为患者，每行为program，颜色表示0–1的相对usage；顶部为癌种标签。患者先按最大usage所属程序排序，再按该最大usage强度排序，因此连续色块部分来自展示排序，不是额外的无监督发现。白线分隔主导程序组。

P后面的星号表示bootstrap稳定性达标，**不表示生存显著性、癌种富集显著性或GSEA显著性**。同一患者可同时使用多个程序；取argmax只是方便展示。P1等编号只在各自cluster内有效，C09-P1与C11-P1没有天然对应关系。

## 每个Cxx目录中的结果
- `summary.json`：主导轴、方向、患者数、癌种数、选定rank及总体指标。
- `selected_feature_genes.tsv`：候选基因与顺序。
- `patient_program_usage.tsv.gz`、`patient_usage.npy`：患者程序使用量；读取NPY必须核对脚本和表中的行列顺序。
- `program_gene_loadings.tsv.gz`、`program_loadings.npy`：程序基因loading，区分正负通道。
- `rank_diagnostics.tsv`、`bootstrap_stability.tsv.gz`、`program_stability.tsv`：rank选择和程序稳定性。
- `program_risk_associations.tsv`：程序与模型风险的关联摘要，不应自动解释为独立临床预后证据。

全体cluster汇总位于`all_cluster_nmf_summary.tsv`。下游富集应使用对应完整候选基因排序，不能只拿展示图里的代表基因；保护状态富集通道与正风险状态不同，见第四步。
