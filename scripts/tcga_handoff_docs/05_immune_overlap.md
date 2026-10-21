# 第五步：跨轴免疫程序的leading-edge对照

## 为什么做这一步
不同状态中的程序都可能被注释为“免疫相关”。本分析区分：它们是否由同一批高loading免疫基因支撑，还是只有功能名称相似。它是对第四步的描述性补充，不是又一次GSEA或独立验证。

## 比较对象和固定通路
四个程序为C09-P4 H+、C11-P4 H+、C03-P1 H−、C03-P5 H−。统一比较三个KEGG Legacy基因集：Chemokine signaling、Cytokine–cytokine receptor interaction、Cell adhesion molecules。不是分别取每个程序最显著的不同通路再求并集。即使原通路不显著也保留其NES/FDR，未测试与测试后无重叠必须区分。

leading-edge是GSEA富集峰对应的核心贡献成员，不等于整条通路所有基因，也不是NMF独立筛出的Top100核心基因。成员来自第四步的原始GSEA记录。

## 两种重叠量怎么算
原始Jaccard：J(A,B)=|A∩B|/|A∪B|，其中A、B是同一条通路在两个程序的leading-edge集合。

共同候选集敏感性：先取两个程序NMF候选基因的交集U，再令A'=A∩U、B'=B∩U，计算J(A',B')。这用于查看不同候选基因覆盖对结果的影响，但不是在统一背景上重跑GSEA，也不是统计校正。并集为空时记NA；共同并集小于5时标记支持太少。

## 附图怎么读
`Supplementary_immune_program_leading_edge_comparison.pdf`每一行是一条固定通路，左列是原始leading-edge Jaccard，右列是共同候选基因范围内的敏感性结果。横纵轴为四个程序；色标0–1，越大代表集合越相似，NA表示不可计算/未测试，不是零。对角线的自我比较不构成复现证据。

颜色必须结合交集和并集数、候选覆盖、NES和FDR一起看。低重叠可能来自不同分子实现，也可能来自候选集差异、统计功效或较小集合；不能只凭一个低Jaccard宣称“机制不同”。高重叠表示共享部分基因成员，也不意味着风险方向或表达活性相同。

## 文件与统计限制
`Supplementary_Table_immune_leading_edge_overlap.tsv`列出两两比较、原始及限制后交并集、精确共享/特有基因和小支持标记。`Supplementary_Table_immune_pathway_NES_FDR.tsv`用于确认原通路是否被测试及其证据等级。`Supplementary_immune_overlap_methods.txt`记录原始方法。

本分析没有对重叠计算超几何P值或新的FDR，也没有重拟合NMF。应报告“共享程度/差异线索”，不要写成“显著不同”，除非另有正式检验。
