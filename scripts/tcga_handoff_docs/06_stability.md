# 第六步：C11调控程序的条件稳定性及表达支持

## 两类证据必须分开
本步骤包含泛癌C11的bootstrap稳定性，以及附图中TCGA-BRCA/C11子集的表达调控活性支持。两者使用不同样本范围；不能把BRCA表达结果描述为在所有C11癌种中成立。

## 泛癌bootstrap怎么做
使用原始722例C11患者，固定原NMF的1,000候选基因和rank=4，进行30次患者有放回抽样。重新拟合NMF（最多400次迭代、容差1e-4），通过Hungarian算法按H余弦相似度匹配回参考P1–P4，不按新模型的原始编号直接比较。

`Supplementary_Table_C11_bootstrap_stability.tsv`的120条记录对应30次×4个程序，包含匹配余弦、迭代和是否触及迭代上限。高余弦支持给定特征集和rank下的loading稳定，但并不检验重新选基因或重新选rank后的稳定性。第三步更一般的bootstrap流程应独立查阅。

进一步对24个regulon在完整H+排序上计算加权signed ES，记录ES符号频率、分位数区间及程序相对其余程序平均ES的差异。`Supplementary_Table_C11_bootstrap_regulon_ES.tsv`有24×4=96条摘要。这些是描述性bootstrap区间，不是重复GSEA的NES置信区间，也不是新BH-FDR，不能用它们替换第四步显著性。

## 表达支持怎么得到
原附图另使用精确ID匹配的233例TCGA-BRCA/C11患者，沿用泛癌程序标签，没有在BRCA中重新拟合P1–P4。输入为完整校正log2(x+1)表达；重复gene symbol取均值，非恒定基因在该子集内z-score。

DoRothEA A/B激活/抑制边分别取+1/−1，丢弃不明确或冲突方向；25个候选TF中24个满足至少15个有符号可测量靶基因，ZNF263因缺乏适用有符号边被排除。使用带截距的ULM斜率t统计量作为相对TF活性分数，非靶基因权重为零。这里才是基于表达的分析，和第四步基于归因排序的TF靶基因富集不同。

程序组对其余患者的OLS比较调整标准化MEred；敏感性模型进一步调整轴Shapley绝对值。每种模型按24×4=96次检验单独BH。连续usage偏Spearman属于另一组敏感性检验。未调整亚型、纯度和完整临床混杂，不是因果验证或独立外部验证。

## 图怎么读与使用边界
`Supplementary_C11_regulatory_stability_expression_support.pdf`中的bootstrap面板展示匹配loading余弦分布，越接近1表示在当前条件下越稳定；表达面板按图例显示相对调控活性及组间支持，不能将其颜色当作GSEA NES。全部面板的范围和编码以原图注为准。

本目录整理表重点是bootstrap；原图涉及的完整表达中间矩阵、患者级ULM分数及模型系数并未全部放入本目录。因此图可供查看，但仅凭当前bootstrap表不能逐项重建表达面板。重跑需原脚本所引用的表达及regulon资源。原方法完整保存在`Supplementary_C11_stability_methods.txt`。
