# 第四步：风险程序的功能与调控注释（Figure 5C）

## 输入及分析对象
输入是第三步NMF的完整候选基因loading排序，而不是原始表达或差异表达统计量。C09/MEpink+、C11/MEred+分析H+；C03/MEblue−分析H−。这里的“完整”指进入该NMF的候选基因排名，并非全基因组。没有将H+和H−相加或相减作为本图的主要排序指标，也不是仅对Top5/Top100做ORA。

## 数据库和结果范围
`Supplementary_Table_GSEA_all_5464.tsv`包含当前整合分析的5,464条结果，保留不显著结果。覆盖PID、KEGG Legacy、KEGG Medicus、WikiPathways、C4:3CA、C4:Cancer Modules、C6扰动签名、DoRothEA A/B、GTRD及C11的CollecTRI。CollecTRI仅测试C11，不能把其他cluster缺失的结果解释为不富集。

这张主表不是全部历史Reactome/GO运行结果的集合。单独的C6、GTRD、CollecTRI完整表是主表子集，不是可相加计算的额外独立检验。不同资源有重叠基因和知识来源，多个数据库命中不等于多次独立验证。

## GSEA及统计字段
预排序GSEA检验某个基因集成员是否集中于loading排序的某一端。NES为标准化富集分数；正NES表示偏向该通道高loading端，负NES表示偏向低端。对于H−排序，正NES表示偏向负风险贡献通道的高loading端，不能翻译成“该通路表达受抑制”。TF靶基因集富集也不等于TF被激活。

`enrichment_score`为原始ES；`nominal_pvalue`为来源名义P；`permutation_pvalue_for_bh`为用于BH的P；正式汇报字段`bh_fdr`按每个程序、每个数据库内部计算。它不是所有数据库联合控制的FDR。`gsea_fdr`、`global_bh_fdr`、`pooled_bh_fdr`如有值，应保留原定义，不能替换或混用。

证据分级：FDR<0.05可进入正式结果筛选，但还需审查解释与冗余；0.05–0.10在有其他支持时作为探索性；0.10–0.20仅用于候选方向；≥0.20不进入候选。FDR<0.20不是“已证实发现”的统一阈值。不能仅因为某个term更符合预期，就把弱统计证据写成确定机制。

## Figure 5C怎么读
按三个状态分面；横轴为程序，纵轴是代表性基因集。标签中的程序表示促使该条目被选作代表的程序，不意味着该通路只可能出现在这一程序。颜色为原始基因集NES，图中点大小固定，不编码FDR。黑边表示BH-FDR<0.05；较浅点用于显示较弱或不显著的上下文，具体透明度见图例。

图中并没有计算一个新的“主题分数”，也没有把CAF、ECM等名称当作新数据库term重做检验。代表条目经事后生物学审阅并考虑冗余，不是预注册选择。`Figure5C_selected_evidence`与`Figure5C_redundancy_audit`表保留选择和重叠审计轨迹；对应脚本是`64_plot_representative_figure5c.py`。

## 解释边界
C6的UP/DN来自原始扰动实验，并不直接对应本队列的激活/抑制或风险方向。C4编号模块应保留原始ID，功能简称只能作为解释性标签。主图可以支持不同程序具有不同的功能/调控靶基因组成，但不证明真实表达活性最高、细胞比例变化或因果调控。归因富集与表达验证必须分开报告；本步骤主GSEA不是基于患者表达算出的ssGSEA。

## 下游建议
先按cluster、program、通道和library定位，再查看精确pathway ID、NES、FDR和leading-edge；避免跨数据库直接按最小FDR给程序命名。保留未注释程序，不强迫每个program对应单一经典通路。需要实验或表达验证时应另外设计，不可将此表直接转换成活性矩阵。
