# RC-EdgeDistill Major Revision 逐条意见闭环修改步骤

**稿件编号：** IJSEKE-D-26-00185  
**题目：** *RC-EdgeDistill: LLM-Supervised Redaction-Consistent Learning for Hospital Feedback at the Edge*  
**期刊：** *International Journal of Software Engineering and Knowledge Engineering*  
**编辑决定：** Major Revision  
**编辑来信日期：** 2026-09-23  
**返修截止日期：** 2026-10-23  
**本文件版本：** 2026-10-08  

> 本文件是“修改执行单”，不是新的论文正文。所有 `[PLACEHOLDER]` 必须在真实材料、真实运行日志或研究团队确认后替换，不能原样提交。方案以实际投稿文件 `RC_EdgeDistill_IJSEKE_Submission.docx` 为主版本；Markdown `IJSEKE-D-26-00185.md` 用于定位原始文字和生成修改草稿。

## 1. 先给出结论：我能否独立完成这次返修

### 1.1 Codex 可以直接完成的工作

以下工作不需要伪造实验结果，可以由我直接完成：

1. 将审稿意见拆成逐条 rebuttal，统一术语、章节编号和贡献叙事。
2. 根据现有稿件改写摘要、引言、方法、实验、讨论、局限、结论和数据可用性声明。
3. 把 LLM 标签流程改写成可复现的方法段落，插入模型、版本、日期、参数和提示词占位符。
4. 生成完整 prompts、JSON schema、裁决规则、数据字典、split manifest 和 supplementary-material 模板。
5. 生成 duplicate/near-duplicate/source-leakage 审计脚本方案、分组划分方案、bootstrap/seed/CI 分析方案和统一计时协议。
6. 按最终真实结果回填表格、绘图说明、结果段和逐条回复，并检查“结果—表格—摘要—结论”数值一致性。
7. 将最终 Markdown 内容整理为 Word 修订稿所需的章节块，并执行文档排版、引用、表格和图形的复核流程。

### 1.2 获得材料后可以完成的工作

只要用户提供原始材料或允许在本地运行现有代码，我可以完成：

- 从 LLM 调用日志提取 provider、model、snapshot/version、调用日期、temperature、top-p、seed、max tokens、system/user prompt 和响应格式；
- 从原始数据和代码生成 exact duplicate、near-duplicate、source-group、synthetic-lineage 审计结果；
- 训练并比较 validation-only 选择的 λ、多个随机种子、语言/来源子集、rare-class 指标和统一 end-to-end latency；
- 对人审标签、PII 挑战集和模型输出计算 Cohen’s κ、Fleiss’ κ 或 Krippendorff’s α、human–LLM agreement、precision/recall/F1、bootstrap CI；
- 生成可放入正文和补充材料的表格、图形和审计日志摘要。

### 1.3 必须由用户/研究团队确认或执行的事项

下列内容不能凭空补写：

1. 实际使用的 LLM provider、精确模型名和版本、调用日期及完整请求参数；
2. 当时真实使用的两套 prompt、JSON schema 和失败重试规则；
3. 冲突裁决者的真实身份/角色、培训方式和裁决记录；
4. 独立人审者的数量、专业背景、盲法和原始标注；
5. 伦理审批、豁免、公开网页收集合规判断和去标识化责任人；
6. 原始数据、来源标识、医院/网站/账号分组信息；
7. 多随机种子重训、统一计时和 PII challenge set 的真实数值。

### 1.4 当前不能诚实生成的内容

不能直接编造 human–LLM agreement、inter-rater agreement、PII false-negative 数量、source-wise F1、多个 seed 的均值/CI、伦理豁免编号、真实 prompt、LLM 版本或新的 latency。若这些材料无法取得，必须在论文中明确写成 limitation，并将结论降格；不能用占位符冒充完成结果。

## 2. 返修总策略和证据门槛

本轮返修的中心论点应由“分类性能提升”改为：**RC-EdgeDistill 在 LLM 辅助弱监督条件下，使脱敏前后语义预测更稳定，并在 CPU 资源约束下保持可用分类性能；该结论受到独立人审、分组划分、跨语言/来源评估和未见 PII 小样本测试的约束。** 现有 0.6896 对 0.6877 的分类差异不能宣称显著优越；98.67% 对 95.33% 的一致性结果才是主要方法性证据。

### 2.1 主修改块索引

| 编号 | 主修改块 | 解决的问题 | 必须形成的证据 |
|---|---|---|---|
| M1 | LLM provenance and prompts | R2-1、R3-1、R3-2、R3-16 | provider/model/version/date/settings/prompts/log hash |
| M2 | Human validation and adjudication | R2-1、R3-3、R3-4 | blind human audit, agreement and adjudication log |
| M3 | Leakage-safe split audit | R2-2、R3-8、R3-16 | manifest, duplicate/near-duplicate/source/lineage report |
| M4 | Language/source/rare-class reporting | R2-2、R3-5、R3-6、R3-7 | Chinese/English/source tables and support counts |
| M5 | Validation-only λ and repeated runs | R2-3、R3-11、R3-12 | locked test protocol, seed table, CI |
| M6 | Independent PII challenge set | R2-4、R3-9、R3-16 | human-annotated challenge set, P/R/F1, FN examples |
| M7 | Changed-record consistency | R2-4 | changed/unchanged denominators and CIs |
| M8 | Baseline tuning and multilingual fine-tuning | R2-5、R3-16 | hyperparameter table and compact multilingual encoder run |
| M9 | Common end-to-end timing | R2-5、R3-13 | hardware/boundary/warm-up/repetition protocol |
| M10 | Claim calibration | R3-10、R3-14 | revised abstract/discussion/conclusion |
| M11 | Ethics and data collection | R3-15 | ethics/exemption/collection/de-identification statement |
| M12 | Reproducibility package | R3-16 | prompts/rules/config/code/manifest/checksum/archive |

### 2.2 建议新增的补充材料文件

1. `Supplementary_A_LLM_Protocol.md`: provider、模型版本、日期、参数、完整 prompts、JSON schema、重试和 adjudication 规则。
2. `Supplementary_B_Data_Split_Audit.csv`: record hash、source group、language、split、synthetic lineage、duplicate cluster。
3. `Supplementary_C_Human_Validation.csv`: 脱敏后的 record ID、人审者、四类标签、confidence、adjudication。
4. `Supplementary_D_PII_Challenge_Set.csv`: PII 类型、是否存在、span、模块输出、错误类型。
5. `Supplementary_E_Training_and_Timing.yaml`: seed、λ、optimizer、hardware、batching、timing boundary。
6. GitHub release/tag 和 DOI/永久归档链接；提交前检查其中没有原始个人信息、账号信息或未授权文本。

## 2.3 基于实际 workspace 的新增审计结论

用户提供的完整本地导出目录为：

`D:\CC前沿分析\在投论文\IJSEKE\workspace-019f6527-2551-7e7b-9e83-c6c40c243e0b`

该目录来自用户所述的 Arena 免费云环境，但当前导出内容实际可见的是 frozen sentence-embedding baseline 子任务，而不是完整 RC-EdgeDistill 主模型归档。详细结果见 [RC_EdgeDistill_workspace-审计报告.md](D:\CC前沿分析\在投论文\IJSEKE\RC_EdgeDistill_workspace-审计报告.md)。这会改变原先“缺少证据”的判断：现在有一部分证据已经可以直接核验，但也发现了一个必须先修复的 split 问题。

已核验事实：

- `prepared_paired_dataset_2000.csv` 有 2,000 行，字段包括 `language`、`source_family`、`split`、paired original/redacted text、三项语义标签、PII 字段和 confidence。
- 现有 split 为 train/validation/test = 1,400/300/300；中文/英文为 1,049/951；test 中中文/英文为 160/140。
- `source_family` 只有 Chinese public web 和 English public hospital reviews 两类，并且与 language 完全一一对应；因此 language effect 和 source effect 不能在本数据中独立识别。
- 对 original 和 redacted 文本做 NFKC、大小写和空白归一化后，各有 38 个完全重复组、85 条重复记录；其中 24 个重复组、56 条记录跨越 split。
- test 中有 142 条 changed records、158 条 unchanged records；该分母已经可以用于 changed-only consistency，但当前 workspace 没有逐条 RC 或 baseline predictions，尚不能产生 changed-only 数值。
- MiniLM、multilingual E5-small 和 BGE-small-zh frozen baselines 均已完成，但都是单 seed、旧 split、LogisticRegression，且代码计时只覆盖 encoder 输出后的 classifier prediction，不是完整 end-to-end latency。

因此，M3 不能再写成“检查是否存在 leakage”；必须写成“发现 exact cross-split duplicates，先重建 split，再重新评估”。M4 必须把 language/source 结果标为混淆诊断。M9 不能直接使用现有 baseline latency。M1、M2、M6、M11、M12 仍需要从 Arena 导出更完整的日志、材料或由研究团队补做。

### 2.4 现有 workspace 能否直接完成哪些修改

| 修改项 | 当前是否能直接完成 | 判断 |
|---|---|---|
| 数据总量、语言、source_family、split 统计 | 可以 | 已从 CSV 核验 |
| changed/unchanged 分母 | 可以 | 已从 paired text 核验 |
| exact duplicate 跨 split 审计 | 可以 | 已发现 24 个跨 split 重复组、56 条记录 |
| near-duplicate/source/lineage 审计 | 不能完整完成 | 缺 near-duplicate 脚本、医院/网站/账号/source group 和 synthetic parent metadata |
| frozen baseline aggregate 结果整理 | 可以 | 三个结果目录含 CSV、manifest、报告 |
| RC 主模型重新训练 | 当前不能 | 目录中没有主模型训练代码和权重/配置 |
| validation-only λ 选择、多 seed | 当前不能 | 缺主模型代码、逐 seed 输出和 locked split |
| human–LLM/inter-rater agreement | 不能 | 没有人审标签和独立审计记录 |
| independent PII challenge set | 不能 | 当前只有规则/合成数据线索，没有独立 gold set |
| 统一 end-to-end timing | 不能直接采用现有结果 | 现有脚本未把 embedding/tokenization/redaction 纳入计时 |
| 完整 LLM provenance/prompts/adjudication | 当前不能从该目录恢复 | 目录中没有 LLM logs 或 prompt archive，仅有任务说明称 labels 是 AI-adjudicated weak supervision |

### 2.5 必须新增的 split 修复块

在执行 M5、M8 和最终 Results 写回之前，新增以下强制步骤：

1. 从现有 2,000 条记录中按规范化 original/redacted text 建立 duplicate clusters；同一 cluster 只能进入一个 split。
2. 在取得 source/hospital/website/account/synthetic lineage 后继续做 group-aware split；缺失 group ID 的记录单独标记。
3. 保留当前 1,400/300/300 作为“submitted split”审计对照，但不把它当作返修确认性结果。
4. 重新生成 `locked_split_manifest.csv`，并在 Supplementary B 报告 exact duplicate、near duplicate、source group 和 synthetic lineage crossing 数量。
5. 只有新 locked test 建立后，才允许选择 λ、训练多 seed、计算语言/source/rare-class/changed-only metrics 和统一 timing。

## 3. 逐条处理 Reviewer #2

## R2-1 LLM 标签有效性、完整流程和独立人审

### 审稿意见

> “The same LLM was used with two instructions, and the final test metrics measure agreement with labels produced through that process. Please report the LLM name and version, complete prompts, generation settings, and the exact adjudication procedure. More importantly, a representative sample—particularly high-priority, privacy/security, medication, and disagreement cases—should be independently reviewed by human evaluators. Human-LLM agreement and inter-rater agreement should be reported.”

### 当前原文与位置

- Markdown 第 124–130 行，`## 3.2 LLM-Assisted Weak-Supervision Label Construction`：`RC-EdgeDistill uses an LLM to construct structured weak-supervision labels...`；`For each record xi, two instructions are denoted by gA and gB...`；冲突时“the source text is checked against the predefined category definitions and dominant-issue rule”。
- Markdown 第 142 行：`The workflow also records a descriptive confidence tag...`。
- Markdown 第 246 行：只报告两套 instruction 的 disagreement rate 和 high/medium confidence 数量。
- Markdown 第 65–67 行：说明这是 LLM-assisted weak supervision，但没有 provider/version/prompt/生成设置。

### 问题判断

当前文字承认是弱监督，却没有让读者复现标签生成，也没有证明测试集标签与独立判断相符。两次 prompt 只能说明 instruction sensitivity，不能替代独立人审。

### 修改步骤

1. 从调用日志恢复 `[PROVIDER]`、`[MODEL_NAME]`、`[MODEL_VERSION_OR_SNAPSHOT]`、`[API_DATE_RANGE]`、temperature、top-p、max tokens、seed、response format、retry policy 和 system prompt。
2. 把两套完整 prompt 原文放入补充材料；正文仅给 prompt 目的、变量和 schema，并注明完整文本见 Supplementary A。
3. 明确定义 `a(·)`：一致时直接保留；不一致时由 `[ADJUDICATOR_ROLE]` 按 category definitions、dominant-issue rule 和 priority/PII rules 作出最终标签；若无法决定，记录 `uncertain`，说明如何处理。
4. 建立独立、盲法人审样本。建议分层抽取：全部 disagreement 案例或至少 `[N_DISAGREE]` 条；high-priority `[N_PRIORITY]`；privacy/security `[N_PRIVACY]`；medication 全部 17 条或可获得数量；其余按语言/来源随机抽取。
5. 至少两名独立人审者分别标注；不向其展示 LLM 标签。第三人只处理冲突，不参与初始独立评分。
6. 报告每个任务的 human–LLM agreement、两位人审者的 inter-rater agreement、95% CI、样本量和缺失/弃权数。
7. 若无法完成人审，不能写“validated”；改为“an audit protocol was prepared but independent validation remains future work”，并相应降低结论。

### 建议新增/替换英文正文

放在 `## 3.2` 现有第一段之后：

```text
Label construction used [PROVIDER] [MODEL_NAME] ([MODEL_VERSION_OR_SNAPSHOT]) through [API_OR_LOCAL_INTERFACE] between [START_DATE] and [END_DATE]. The recorded generation settings were temperature = [TEMPERATURE], top-p = [TOP_P], maximum output tokens = [MAX_TOKENS], seed = [SEED_OR_NOT_SUPPORTED], and response format = [JSON_SCHEMA_OR_TEXT]. The complete system and user prompts, category definitions, output schema, retry policy, and a checksum of the labeling log are provided in Supplementary A. No label-generation call was made during routine local inference.

For each record, the two prompt outputs were parsed against the same schema. If all task fields agreed, the agreed vector was retained. If any field differed, [ADJUDICATOR_ROLE] reviewed the source record against the predefined category definitions and dominant-issue rule, recorded the reason code, and selected one final label; unresolved cases were assigned [UNCERTAIN_POLICY]. The adjudicator did not see the human-audit labels during this procedure. This operation is procedural adjudication, not a learned model.
```

放在 `## 4.1` 标签统计之后：

```text
To assess label validity independently, we sampled [N_AUDIT] records using a prespecified stratified protocol covering [N_DISAGREE] disagreement cases, [N_PRIORITY] high-priority cases, [N_PRIVACY] privacy/security cases, [N_MEDICATION] medication cases, and language/source strata. Two reviewers independently labeled the records without access to the LLM outputs. Human–LLM agreement was [AGREEMENT_METRIC] = [VALUE] (95% CI [LOW, HIGH]) for [TASKS], while inter-rater agreement was [IRR_METRIC] = [VALUE]. The adjudicated human labels and sampling counts are provided in Supplementary C. These results qualify the main test metrics as performance against an LLM-assisted label schema with an independent audit, rather than as a substitute for full clinical annotation.
```

### 修改后放置位置

- 方法细节：`## 3.2`，紧接当前第 126–130 行后。
- 审计设计与数值：`## 4.1`，紧接当前第 246 行后。
- 完整 prompt/schema/log：Supplementary A；人审原始表：Supplementary C。
- 摘要、讨论和结论只保留审计后的总体结论，不塞入过长流程。

### 必须提供/实际运行

真实 LLM 日志、prompt 文件、裁决者信息、独立人审标签、审稿者资质/培训、盲法记录和 agreement 计算。Codex 可生成抽样脚本、标注表、统计脚本和英文段落，但不能替研究团队制造人审。

### 完成验证标准

读者能够仅凭正文+Supplementary A 重现一次标签调用；每个审计样本有可追溯 ID；两位人审者的原始标签和最终裁决可核对；agreement 指标、CI、分层样本量和缺失数相加一致；论文不再把两次 LLM prompt 称作两名 annotator。

### 对应 rebuttal

```text
We agree that two prompt passes do not constitute independent human annotation. We added the exact provider/model/version, labeling dates, generation settings, complete prompts, schema, retry policy, and adjudication rule in Section 3.2 and Supplementary A. We also added a blinded stratified human audit covering disagreement, high-priority, privacy/security, medication, language, and source strata, and report human–LLM and inter-rater agreement in Section 4.1 and Supplementary C. The manuscript now explicitly describes the main metrics as agreement with an LLM-assisted label schema qualified by an independent audit.
```

## R2-2 数据划分、泄漏、语言与来源

### 审稿意见

> “Please provide more information about the train, validation, and test split. It should be confirmed that duplicate or near-duplicate records, reviews from the same source, and synthetic variants of the same record did not appear across different splits. Since the dataset combines Chinese and English records from different sources, separate results by language and source would help determine whether the model performs consistently or mainly learns source-specific patterns.”

### 当前原文与位置

- Markdown 第 240–242 行：`The dataset contains 2,000 records...`；`The training, validation, and test sets contain 1,400, 300, and 300 records...`；只说明 synthetic templates appended to training records，验证/测试没有新增 synthetic identifiers。
- Markdown 第 246–248 行：语言和类别总数，但没有 source-wise 结果、duplicate cluster 或 near-duplicate audit。
- Table 4（约第 263–275 行）：只有总量和 split 摘要。

### 问题判断

“没有向验证/测试加入合成标识符”不等于证明原始记录没有 exact/near duplicate、同一来源泄漏或训练记录的 synthetic variant 泄漏到其他 split。跨语言总分也可能被来源差异掩盖。

### 修改步骤

1. 为每条记录建立不可逆 record hash、source ID、website/hospital/account group、language、synthetic parent ID 和 duplicate cluster ID。
2. 运行 exact duplicate（规范化文本 hash）、near-duplicate（字符 n-gram/Jaccard 或 MinHash，阈值 `[NEAR_DUP_THRESHOLD]`）和 source-group leakage 审计。
3. 重新划分：先按 source/hospital/website group 分组，再在组内按语言和标签分层；同一原始记录及其所有 synthetic variants 必须进入同一 split。
4. 生成公开的 split manifest，但不暴露原始文本；报告各 split 的语言、来源、类别和 synthetic lineage 数量。
5. 对最终锁定测试集报告 Chinese/English 和每个 source；若某 source 太小，报告 support、CI 并标记 exploratory。
6. 若原数据缺少 source ID，不能声称 source-level split 已完成，应将结果称为 language-wise exploratory analysis，并在 limitation 中说明。

### 建议新增英文正文

替换 `## 4.1` 当前第 242 行后半段：

```text
Before splitting, records were normalized and audited for exact duplicates, near duplicates, shared source groups, and synthetic lineage. Exact duplicates were identified by [NORMALIZATION_AND_HASH], and near duplicates by [NEAR_DUPLICATE_METHOD] with threshold [THRESHOLD]. All records belonging to one duplicate cluster, source group, or synthetic parent lineage were assigned to a single split. The resulting group-aware split contains [N_TRAIN], [N_VAL], and [N_TEST] records, with [N_GROUP_TRAIN], [N_GROUP_VAL], and [N_GROUP_TEST] independent groups, respectively. No duplicate cluster, near-duplicate pair above the prespecified threshold, source group, or synthetic lineage crosses the split boundary. The anonymized split manifest is provided in Supplementary B.

Because the Chinese and English subsets originate from different sources, all main metrics were additionally computed by language and by source. These subgroup estimates are reported with their test support and confidence intervals in Table [LANG_SOURCE_TABLE]. They are interpreted as transportability diagnostics rather than evidence of clinical cross-country validation.
```

新增结果表脚注：

```text
Subgroup results are evaluated on the locked test partition. A dash indicates that a subgroup has insufficient support for a stable estimate. Confidence intervals are paired bootstrap intervals within the subgroup.
```

### 修改后放置位置

`## 4.1` 数据集段落；新增 Table 4a（split audit）和 Table 5a（language/source results）；Supplementary B 保存逐记录 manifest；`## 5.3` 讨论来源偏差。

### 必须提供/实际运行

原始 source metadata、规范化规则、duplicate/near-duplicate 运行结果、重新划分后的 manifest、各组测试预测。Codex 可写审计脚本和生成表格，但无法在缺少 source metadata 时证明没有 source leakage。

### 完成验证标准

所有 synthetic variants 与 parent 同 split；exact/near duplicate cluster 跨 split 数为 0；source-group 跨 split 数为 0；train/validation/test 数量和 Table 4 相符；中文+英文及每个 source 的 support 总和等于测试集 300（或在排除无 source 的记录时明确说明）。

### 对应 rebuttal

```text
We added a pre-split duplicate, near-duplicate, source-group, and synthetic-lineage audit and regenerated the partitions using group-aware assignment. The anonymized manifest is in Supplementary B. We also added locked-test Chinese/English and source-wise results with subgroup support and confidence intervals. Where source metadata were unavailable, we explicitly label the corresponding analysis exploratory and avoid claiming source-level generalization.
```

## R2-3 λ 选择、多随机种子和置信区间

### 审稿意见

> “The model-selection procedure needs clarification. The manuscript states that several values of the consistency weight were examined and that λ = 0.10 performed better on the test set, although λ = 0.05 was retained as the main model. Hyperparameters should be selected using only the validation set, followed by a single evaluation on a locked test set. Results from multiple random seeds should also be reported, together with confidence intervals for the main task-level metrics.”

### 当前原文与位置

- Markdown 第 398 行：`We evaluated λ ∈ {0,0.03,0.05,0.10,0.25}... In a post-hoc test-set analysis, λ = 0.10 reached...`
- Markdown 第 329–331 行：主结果只有当前一次训练/测试及 2,000 次 bootstrap；没有多 seed 分布。
- Markdown 第 244 行：当前模型使用 random seed 20260715。

### 问题判断

当前做法透明地承认了 test-set post-hoc，但仍不能作为严格 model-selection protocol。必须把 λ 选择封闭在 validation，锁定 test，并将 λ=0.10 的旧结果重新标为历史敏感性结果或删除。

### 修改步骤

1. 先完成 M3 的 group-aware split；禁止在重新划分后继续使用旧 test 结果选参。
2. 预先定义 validation selection rule，例如最大化 `validation composite macro-F1`，若差异在 `[TIE_TOLERANCE]` 内则选择 mean consistency 更高者，再以较小 λ 破平。
3. 在 seeds `[SEED_LIST]` 上仅用训练集训练、validation 选择 λ；锁定 λ 后，所有 seed 在 test 上只评估一次。
4. 报告每个 seed 的 macro-F1、三个任务 F1、consistency、changed-record consistency、PII precision/recall/F1 和 latency；给出 mean、SD、95% CI。
5. 重新决定是否保留 λ=0.10 旧 post-hoc 数字。建议删除旧的“both above the main model”，改为说明旧版本探索结果不参与新分析。
6. bootstrap 应在锁定模型的测试预测上进行；seed CI 与 paired bootstrap CI 分开命名。

### 建议替换英文正文

替换 `## 4.5` 当前第 398–400 行：

```text
The consistency weight was selected without using the test set. We evaluated λ ∈ {0, 0.03, 0.05, 0.10, 0.25} on the training/validation protocol using [SEED_SELECTION_PROTOCOL]. The prespecified selection criterion was [VALIDATION_CRITERION], with [TIE_BREAK_RULE]. This procedure selected λ = [SELECTED_LAMBDA] before the test partition was opened. We then locked the selected configuration and evaluated each of [N_SEEDS] random seeds once on the locked test set. The earlier λ = 0.10 test-set comparison was not used for model selection and is [REMOVED/RETAINED ONLY AS A HISTORICAL EXPLORATORY NOTE].

Across seeds, the locked-test composite macro-F1 was [MEAN] ± [SD] and mean redaction consistency was [MEAN] ± [SD]. The 95% seed-based confidence intervals were [CI], and paired bootstrap intervals for the locked test predictions are reported in Table [TABLE_ID].
```

### 修改后放置位置

`## 4.5` λ 选择与重复实验；`## 4.3` 主结果表改为 mean ± SD/CI；`## 5.1` 讨论稳定性；Supplementary E 给出每个 seed 的日志。

### 必须提供/实际运行

重新训练日志、validation scores、seed list、固定硬件/软件版本、锁定 test 的预测文件。Codex 可写运行和统计脚本，但不能填入未经运行的均值和 CI。

### 完成验证标准

提交记录能显示 test 文件在 λ 选择时未被读取；λ 的选择可由 validation 表独立复算；每个 seed 有完整输出；主表、摘要和结论使用同一锁定 test 统计量。

### 对应 rebuttal

```text
We revised the selection protocol so that λ is chosen exclusively from the validation partition under a prespecified criterion. The selected configuration was then evaluated once on a locked test set across [N_SEEDS] seeds. We report per-seed results and confidence intervals, and no longer use the earlier post-hoc λ = 0.10 test comparison as evidence for model selection.
```

## R2-4 PII 评估和 changed-record consistency

### 审稿意见

> “A macro-F1 of 1.000 for the deterministic module appears to reflect agreement with labels and synthetic patterns based on the same predefined rules. This does not demonstrate reliable detection of unseen names, addresses, institutional identifiers, spelling variations, or more complex personal information. Please evaluate the module on an independently annotated set and report precision, recall, and false-negative examples. Redaction consistency should also be calculated separately for records that were actually changed by redaction, since unchanged records can make the overall consistency score appear higher.”

### 当前原文与位置

- Markdown 第 204 行：PII 模块范围是 placeholders、email、URL、Chinese mobile/ID、telephone、WeChat/QQ handles。
- Markdown 第 384 行：`macro-F1 reached 1.0000... This value reflects agreement within the existing placeholders and predefined patterns.`
- Markdown 第 386 行：2,100 injected patterns 全部检测，但明确是 configured rule set。
- Markdown 第 305–311 行：consistency 分母是全部 `N`，没有 changed/unchanged 分层。

### 问题判断

当前限制说明是好的，但审稿人要求新增独立证据。synthetic stress test 与规则同源，不能评价未见实体；总体 consistency 可能因大量 unchanged records 被抬高。

### 修改步骤

1. 建立独立 PII challenge set，来源至少包括人工编写/独立抽样的 Chinese/English names、地址、机构标识、拼写变体、电话号码、账号、时间地点组合和复杂上下文；不得由同一规则生成器自动标注后直接作为 gold label。
2. 由两名独立标注者标注 PII span/type，记录 adjudication；报告每类 support、precision、recall、F1 和 false-negative examples（脱敏示例）。
3. 将现有 2,100 synthetic patterns 单独标为 “configured-pattern stress test”，不与 independent challenge set 合并成一个 F1。
4. 对每条测试记录计算是否发生真实文本替换：`changed = 1[redacted_text != original_text]`。分别报告 changed、unchanged 和 all records 的 consistency，并报告三个语义任务的分母。
5. 对 changed-only consistency 做 paired bootstrap CI；如果 changed 样本过少，报告 exact count 并明确不稳定。

### 建议替换英文正文

替换 `## 4.4` 当前第 384–386 行：

```text
The deterministic module obtained [SYNTHETIC_SCORE] on the configured-pattern stress test, which contains the same seven pattern families represented by the current rules. This result verifies implementation coverage for those patterns only. It is not evidence of general detection of unseen names, addresses, institutional identifiers, spelling variations, or contextual personal information.

We therefore evaluated an independently annotated PII challenge set of [N_PII] records containing [PII_TYPES] in Chinese and English. Two annotators labeled PII spans and types without access to the module output; disagreements were resolved by [ADJUDICATOR_ROLE]. On this challenge set, precision, recall, and F1 were [P], [R], and [F1], respectively, with [N_FN] false negatives. Representative false negatives, shown in de-identified form, are listed in Supplementary D. The synthetic and independent results are reported separately.
```

替换 `## 4.4` consistency 定义之后的结果段：

```text
We report redaction consistency for all records and separately for records whose text changed under the local redaction function. Of the locked test set, [N_CHANGED] records changed and [N_UNCHANGED] did not. Consistency on changed records was [C_CHANGED] (95% CI [LOW, HIGH]), compared with [C_UNCHANGED] on unchanged records and [C_ALL] overall. The changed-record estimate is the primary diagnostic for whether identifier removal alters semantic predictions; the all-record value is retained for comparability with prior reporting.
```

### 修改后放置位置

PII challenge set和synthetic分开写在 `## 4.4`；changed-only 公式放在 `## 4.2` consistency metric 后，数值放在 `## 4.4`；Supplementary D 放完整 challenge set 和 FN。

### 必须提供/实际运行

独立 PII 样本、两位标注者、span/type gold labels、错误示例、原始/脱敏文本差异标记、changed/unchanged prediction files。不能用现有 synthetic labels 代替独立 gold。

### 完成验证标准

正文明确区分 configured-pattern 与 independent challenge；每个 PII 类型的 support 加总正确；false negatives 可从 Supplementary D 追溯；changed + unchanged = all；changed-only consistency 的分母和 bootstrap 方法清晰。

### 对应 rebuttal

```text
We separated the configured-pattern stress test from an independently annotated PII challenge set and now report precision, recall, F1, support, and de-identified false-negative examples. We also added changed-record and unchanged-record consistency, with the changed-record estimate treated as the primary redaction-stability diagnostic.
```

## R2-5 基线细节、多语言编码器和统一端到端效率

### 审稿意见

> “Please provide the hyperparameters and tuning procedures for all classifiers and confirm that they used the same training and test partitions. A fine-tuned compact multilingual encoder would provide a stronger comparison than frozen embeddings alone. End-to-end latency should also be measured consistently across the most important baselines, including tokenization, feature construction, redaction, inference, and decoding.”

### 当前原文与位置

- Markdown 第 287 行：列出 logistic regression、LinearSVM、SGD-Huber、ridge、ComplementNB、frozen MiniLM/E5/BGE，但没有完整 hyperparameter/tuning 表。
- Markdown 第 329–350 行：主比较表和 model-level latency；第 410 行承认 frozen embedding timing 不统一。
- Markdown 第 420 行：只有 RC-EdgeDistill 完整 local path P50/P95/throughput。

### 问题判断

当前表把不等边的 model-level latency 放在一起，且 frozen encoder 没有 fine-tuning 对照，无法公平支持效率或性能结论。

### 修改步骤

1. 固定 M3 的同一 group-aware train/validation/test split；所有 baseline 使用相同 split、标签和 redacted-view evaluation boundary。
2. 为每个 baseline 记录搜索空间、validation selection、最终参数、随机种子、特征拟合范围和 early stopping；完整表放 Supplementary E。
3. 增加 compact multilingual encoder fine-tuning baseline，例如 `[ENCODER_NAME]`，在训练集微调、validation 选 epoch/lr/batch size、锁定 test 评估；不能只报告 frozen embedding。
4. 统一 end-to-end timing：同一 CPU、线程数、Python/torch/sklearn 版本、batch/streaming 设置、warm-up 次数和重复次数；边界包含 input read/normalization、redaction、tokenization/feature construction、encoder/inference、classifier、decoding。
5. 将旧 model-level latency 单列为 diagnostic，不与新的 end-to-end latency 混合排序。

### 建议新增英文正文

放在 `## 4.2` baseline 列表之后：

```text
All baselines used the same group-aware train, validation, and locked-test partitions and the same redacted-view evaluation protocol. Hyperparameters were selected on the validation set only. The search spaces and selected values for each classifier, vectorizer, frozen encoder, and fine-tuned compact multilingual encoder are listed in Supplementary E. The fine-tuned encoder was [ENCODER_NAME], initialized from [CHECKPOINT], trained with [LR/BATCH/EPOCH/WEIGHT_DECAY], and selected by [VALIDATION_CRITERION].
```

替换 `## 4.5` 当前第 410、420 行有关速度的叙述：

```text
For an apples-to-apples efficiency comparison, end-to-end latency included input normalization, local redaction, tokenization or feature construction, model inference, classifier decoding, and output serialization. All methods were benchmarked on [CPU_MODEL] with [THREAD_SETTING], [SOFTWARE_VERSIONS], [WARMUP] warm-up runs, and [N_REPEATS] measured repetitions using the same record order and batch policy. We report P50, P95, throughput, serialized artifact size, and peak RSS in Table [TIMING_TABLE]. Model-only latency is retained separately as a diagnostic and is not used to rank complete pipelines.
```

### 修改后放置位置

`## 4.2` 调参和新增 baseline；`## 4.5` timing protocol；Table 5 增加 fine-tuned encoder；新增 Table 5b end-to-end timing；Supplementary E 保存配置。

### 必须提供/实际运行

baseline 代码、validation search logs、compact multilingual encoder checkpoint/训练日志、统一硬件计时原始结果。Codex 可以改写和整理，但不能把 frozen embedding 的旧时间当作新端到端结果。

### 完成验证标准

每个方法的训练/验证/测试 split 相同；最终超参数可由 validation logs 复算；新 baseline 有实际 test predictions；timing 表中每个方法都包含同一处理边界，且 P50/P95/throughput 与重复次数一致。

### 对应 rebuttal

```text
We added complete baseline search spaces and selected hyperparameters, confirmed the common leakage-safe partitions, included a fine-tuned compact multilingual encoder baseline, and reran the efficiency comparison under a common end-to-end boundary covering redaction, tokenization/feature construction, inference, and decoding. Model-only timings are now reported separately.
```

## 4. 逐条处理 Reviewer #3

> R3 的 1–4 与 R2-1 重合，5–9 与 R2-2/R2-4 重合，11–13 与 R2-3/R2-5 重合。为避免两处出现互相矛盾的文字，下面每条仍单独闭环，但以 M1–M12 为唯一实施来源。

## R3-1 LLM details

### 审稿意见

> “Please provide the exact LLM name, version, provider, settings, and date used to create the labels.”

### 当前原文与位置

Markdown 第 124–126、246 行的原文摘录为：

```text
RC-EdgeDistill uses an LLM to construct structured weak-supervision labels for the four tasks. The LLM participates in model development but does not process new feedback during routine inference.
For each record xi, two instructions are denoted by gA and gB.
Every record was processed by the two-instruction LLM labeling and checking workflow.
```

这些句子没有给出 exact model/provider/date/settings。

### 修改细节与英文文字

执行 M1。在 `## 3.2` 插入：

```text
The labels were generated with [PROVIDER] [MODEL_NAME], version/snapshot [MODEL_VERSION_OR_SNAPSHOT], on [DATES]. Generation used temperature [TEMPERATURE], top-p [TOP_P], maximum output tokens [MAX_TOKENS], [SEED_OR_NOT_SUPPORTED], and [RESPONSE_FORMAT]. These values were recorded before parsing and are archived in Supplementary A.
```

### 放置与闭环

放在 `## 3.2` 首段之后；Supplementary A 附原始配置和 log checksum。必须由用户提供真实值；验证标准是配置文件、日志和正文逐项一致。回复句：`We added the exact provider, model snapshot, dates, settings, and an archived configuration record.`

## R3-2 完整 labeling instructions

### 审稿意见

> “The paper mentions two different LLM instructions, but the exact prompts are not provided. Please include both instructions in the paper or supplementary material.”

### 当前原文与位置

Markdown 第 126 行的原文为：

```text
For each record xi, two instructions are denoted by gA and gB. The first emphasizes the primary service issue, whereas the second applies a more risk-conservative reading to priority and personal-information risk. These are two passes within the same LLM-assisted labeling workflow, not two human annotators or two separately trained teacher models.
```

该段是功能概括，不是可复现的逐字 prompt。

### 修改细节与英文文字

执行 M1。在 `## 3.2` 增加：

```text
The two complete prompts are reproduced in Supplementary A. Prompt A instructs the model to identify the single dominant service issue using the category definitions and to assign sentiment, priority, and personal-information risk. Prompt B applies the same schema but explicitly asks for a risk-conservative interpretation of priority and personal-information cues. Both prompts require one JSON object with the fields [SCHEMA_FIELDS], a category value from the closed label set, and no explanatory text outside the object. The wording, ordering, and delimiters used in the original labeling run are preserved in Supplementary A.
```

### 放置与闭环

正文给目的，Supplementary A 给逐字 prompt、schema 和 examples。不能根据记忆重写；必须从日志或版本控制恢复。验证标准是脱离论文正文也能复现 API 请求。

## R3-3 Human validation

### 审稿意见

> “The test labels are based on the same LLM assisted labeling process used for training. Please validate at least part of the test set with independent human reviewers.”

### 当前原文与位置

Markdown 第 134 行的原文为：

```text
The student is trained against this target, and the test metrics measure agreement with the same label schema.
```

这句话直接说明测试指标对齐的是同一 LLM 标签 schema；当前没有独立人审样本。

### 修改细节与英文文字

执行 M2。在 `## 4.1` 增加：

```text
We independently reviewed [N_AUDIT] locked-test records. The sample was stratified by [STRATA]. Reviewers were blinded to the LLM labels and model predictions, used the same written category definitions, and labeled all applicable tasks independently. The audit procedure, reviewer training, exclusions, and adjudication log are described in Supplementary C.
```

### 放置与闭环

需要真实人审和 blind protocol；若只能一名审者，须明说并把 inter-rater agreement 标为 not estimable。完成标准：抽样比例、ID、标签、审计者和最终裁决可追溯。

## R3-4 Adjudication process

### 审稿意见

> “When the two LLM outputs disagree, the paper says the source text is checked against the rules. Please explain who performs this check and how the final label is selected.”

### 当前原文与位置

Markdown 第 130 行的原文为：

```text
Agreement is retained directly. When they differ, the source text is checked against the predefined category definitions and dominant-issue rule. The notation a(·) denotes this procedural adjudication step; it is not a learned function.
```

其中没有说明 who performs the check、如何记录 reason code 或无法决定时如何处理。

### 修改细节与英文文字

执行 M2。在 `## 3.2` 替换该句为：

```text
For disagreements, [ADJUDICATOR_ROLE] independently re-read the source record and applied the written category definitions, the single-dominant-issue rule, and the priority/PII decision rules. The adjudicator recorded one reason code and selected one final label; if the rules did not determine a label, the record followed [UNCERTAIN_POLICY]. The adjudicator's decision was made before the independent human audit and was not changed to improve model metrics.
```

### 放置与闭环

用户必须确认真实裁决者、是否逐条记录 reason code、uncertain policy。完成标准是随机抽查 disagreement case 可还原最终标签。

## R3-5 Rare classes

### 审稿意见

> “Some service categories are very small. For example, the medication category contains only 17 records. Please report test counts and per class results for these rare categories.”

### 当前原文与位置

Markdown 第 248 行的原文为：

```text
Clinical process, other/uncertain, and staff communication contained 582, 445, and 318 records, respectively, whereas medication contained only 17. Counts for the remaining categories ranged from 74 to 185.
```

Table 5（Markdown 第 339–350 行）只报告 `Service-issue F1` 汇总列，没有各类别 test support、precision、recall 和 per-class F1。

### 修改细节与英文文字

执行 M4。新增 Table 5a，列 `Task / Class / Test support / Precision / Recall / F1 / 95% CI`。在 `## 4.3` 加：

```text
Because macro-F1 gives equal weight to classes with very different support, we report per-class precision, recall, F1, and locked-test support in Table [RARE_CLASS_TABLE]. Medication has [N_MEDICATION_TEST] test records. Estimates for classes with small support are accompanied by bootstrap intervals and are interpreted descriptively rather than as stable population estimates.
```

### 放置与闭环

必须从锁定 test predictions 计算，不能从全数据 17 条直接推断 test support。完成标准是所有 service classes support 加总为 test support，medication 的 TP/FP/FN 可复算。

## R3-6 Language-wise results

### 审稿意见

> “Please report performance separately for Chinese and English data to show whether the model works similarly for both.”

### 当前原文与位置

Markdown 第 240、242 行的相关原文为：

```text
The dataset contains 2,000 records with LLM-assisted weak-supervision labels: 1,049 de-identified Chinese public-web records and 951 English public hospital reviews.
The validation set includes 158 Chinese and 142 English records, and the test set includes 160 Chinese and 140 English records.
```

第 329–350 行的主结果却只给混合测试集指标，没有按语言拆分。

### 修改细节与英文文字

执行 M4。在 `## 4.3` 增加：

```text
To assess language robustness, we evaluated the locked test partition separately for Chinese and English records. Chinese results were [METRICS] on [N_CN_TEST] records, and English results were [METRICS] on [N_EN_TEST] records. The subgroup intervals overlap/do not overlap [STATE ONLY AFTER COMPUTATION], so the results are interpreted as language-specific diagnostics rather than a formal equivalence test.
```

### 放置与闭环

新增 language/source table 和 per-language consistency；不能只拆 macro-F1，至少拆三项 task F1、consistency 和 PII metrics。完成标准：Chinese+English support 等于 test total。

## R3-7 Source differences

### 审稿意见

> “The Chinese and English data come from different sources and countries. Please discuss whether the model may be learning source specific patterns instead of general hospital feedback patterns.”

### 当前原文与位置

Markdown 第 240 行的原文为：

```text
The Chinese subset comprises hospital-related public-web excerpts, including direct comments and other public accounts. The English subset was obtained after duplicate removal from version 1 of the Hospital Reviews Dataset by junaid6731. According to its Kaggle documentation, the reviews were collected from Google Maps pages for hospitals in Bengaluru, India.
```

Markdown 第 456 行的原文为：

```text
Hospital-owned complaint systems may differ in wording, severity distribution, and institutional terminology.
```

当前没有 source-wise test、source-group split 或 lexical confound audit。

### 修改细节与英文文字

执行 M3/M4。在 `## 5.3` 替换/扩展：

```text
The language subsets are also source subsets: Chinese records came from [CHINESE_SOURCE_DESCRIPTION], whereas English reviews came from the Bengaluru hospital-review dataset. A source-group audit and source-wise locked-test results are therefore required to distinguish feedback semantics from source-specific vocabulary, formatting, and country context. The present data support a cross-source diagnostic, not a claim of country-invariant hospital deployment. Source-specific generalization remains limited where source groups are small or metadata are incomplete.
```

### 放置与闭环

须提供 source IDs 和 source-wise result；若只有两个不可分离 source，明确这是 confounded language/source comparison。

## R3-8 Train and test split

### 审稿意见

> “Please explain whether records from the same hospital, website, or source can appear in both training and test sets. A source level split would give stronger evidence of generalization.”

### 当前原文与位置

Markdown 第 242 行的原文为：

```text
The training, validation, and test sets contain 1,400, 300, and 300 records. The validation set includes 158 Chinese and 142 English records, and the test set includes 160 Chinese and 140 English records. Synthetic templates were appended to training records with a configured probability of 0.35 to increase coverage of common identifier patterns.
```

该段没有说明同一 hospital、website、account 或 source 是否跨 split。

### 修改细节与英文文字

执行 M3。在 `## 4.1` 增加：

```text
Hospital, website, account, and source identifiers were used as grouping variables where available. A group was assigned to one partition only; records lacking a reliable group identifier were flagged and are reported separately. The final audit found [N_CROSS_GROUP] source-group crossings, [N_DUP_CROSS] duplicate crossings, and [N_NEAR_DUP_CROSS] near-duplicate crossings across partitions.
```

### 放置与闭环

若任一 crossing > 0，必须重新划分或从 claim 中明确排除；不能只写“we checked”。Supplementary B 要有 anonymized group IDs。

## R3-9 PII detection

### 审稿意见

> “The local rule module reaches perfect performance on the current labels and injected patterns, but this does not show that it can detect all real names, addresses, or identifiers. Please add a small human checked PII test set if possible.”

### 当前原文与位置

Markdown 第 384–386 行的原文为：

```text
With the deterministic local module, macro-F1 reached 1.0000 on both views against the current dataset labels. This value reflects agreement within the existing placeholders and predefined patterns. It does not imply complete detection of every real-world form of personal information.
The local module detected and fully redacted all 2,100 injected patterns. This controlled result verifies the configured rule set. It is not a claim of general named-entity or address recognition.
```

该限制性说明是正确方向，但当前没有 human-checked independent set。

### 修改细节与英文文字

执行 M6。新增：

```text
The independent PII set was designed to include unseen lexical forms and contextual variants rather than repetitions of the regular-expression templates. Its construction, annotation guide, inter-rater agreement, and error taxonomy are reported in Supplementary D. We do not extrapolate configured-pattern coverage to unrestricted named-entity or address recognition.
```

### 放置与闭环

放在 `## 4.4` synthetic stress test 后；结果必须报告 P/R/F1 和 FN。不能使用原数据隐含标签作为独立 gold。

## R3-10 Model comparison and contribution focus

### 审稿意见

> “RC-EdgeDistill and LinearSVM have very similar classification results, and the bootstrap interval includes zero. Please keep the main contribution focused on redaction consistency rather than classification improvement.”

### 当前原文与位置

Markdown 第 329–333、352 行的原文为：

```text
RC-EdgeDistill with λ = 0.05 ... achieved a composite macro-F1 of 0.6896. LinearSVM was the closest CPU baseline at 0.6877.
The interval includes zero, so these data do not establish a classification-performance advantage over LinearSVM.
RC-EdgeDistill exceeded this value by 0.0984, corresponding to a 16.64% relative improvement under Equation (23).
```

但摘要第 37 行、贡献第 47 行和结论第 482 行仍突出“highest”或“16.64% improvement”。

### 修改细节与英文文字

执行 M10。摘要中将相关句替换为：

```text
On the locked public/de-identified evaluation, RC-EdgeDistill achieved classification performance comparable to the strongest CPU baseline, while its primary measurable advantage was higher redaction consistency: [NEW_MEAN] versus [BASELINE_MEAN]. The classification difference was [DIFF] with 95% CI [CI], which includes zero.
```

`## 5.1` 增加：

```text
The evidence does not establish a classification-performance advantage over LinearSVM. The contribution is instead the explicit redaction-consistency objective and its measurable stability benefit under the evaluated protocol, together with a locally inspectable CPU implementation.
```

### 放置与闭环

删除或降格“exceeded”作为主 claim；16.64% 只能作为 frozen-embedding descriptive comparison，并附“under this dataset and settings”。验证标准：摘要、引言、讨论、结论不再把非显著差异写成 superiority。

## R3-11 Repeated experiments

### 审稿意见

> “Most results appear to come from one train, validation, and test split. Please consider multiple random seeds or repeated splits, especially because the dataset contains only 2,000 records.”

### 当前原文与位置

Markdown 第 244 行的原文为：

```text
It was trained for 160 full-batch epochs using Adam, a learning rate of 0.006, weight decay of 10−4, and random seed 20260715.
```

主结果和速度结果没有多个 seed 的变异性报告。

### 修改细节与英文文字

执行 M5。在 `## 4.3` 增加：

```text
To quantify training stochasticity, we repeated model fitting with [N_SEEDS] prespecified seeds while keeping the locked group-aware partitions fixed. We report each seed, the mean, standard deviation, and 95% interval for macro-F1, per-task F1, overall consistency, changed-record consistency, and PII metrics. Repeated seeds quantify optimization variability; they do not replace an external validation cohort.
```

### 放置与闭环

若采用 repeated splits 而非 fixed test，必须明确每次 split 的 test 不再是同一个 locked test，并相应改 CI 定义。优先固定 locked test + multiple seeds。

## R3-12 Lambda selection

### 审稿意见

> “Please explain how the main value of λ = 0.05 was selected without using the test set. The paper correctly notes that λ = 0.10 was examined only after looking at the test results.”

### 当前原文与位置

Markdown 第 398 行的原文为：

```text
We evaluated λ ∈ {0,0.03,0.05,0.10,0.25} and fixed the reported main model at λ = 0.05 with the local personal-information module. In a post-hoc test-set analysis, λ = 0.10 reached a composite macro-F1 of 0.6934 and consistency of 0.9900, both above the main model.
```

该段明确承认 test-set post-hoc，但没有提供 validation-only selection evidence。

### 修改细节与英文文字

执行 M5，使用 R2-3 的替换段。若新 validation 仍选 0.05，可写：

```text
Under the prespecified validation-only criterion, λ = 0.05 was selected before the locked test predictions were generated. The post-hoc λ = 0.10 result from the submitted version was removed from the confirmatory comparison and is not used in the revised claim.
```

### 放置与闭环

只在真实 validation log 支持时写“selected”；否则写 `[SELECTED_LAMBDA]` 并暂停提交。验证标准：git/log 时间或运行记录证明 test 未参与 selection。

## R3-13 Speed comparison

### 审稿意见

> “The full local pipeline timing is available for RC-EdgeDistill, but the same complete timing is not available for all baselines. A common end to end timing setup would make the efficiency comparison stronger.”

### 当前原文与位置

Markdown 第 410 行的原文为：

```text
Frozen-embedding timings did not uniformly include tokenization, sentence encoding, and feature construction. The recorded values therefore do not support a full end-to-end speed ranking.
```

Markdown 第 420 行只对 selected model 报告 `P50 latency was 4.56 ms per record, P95 latency was 9.73 ms, and mean throughput was 191 records per second`。

### 修改细节与英文文字

执行 M9。替换为统一 protocol 段：

```text
The revised benchmark uses one end-to-end boundary for every method: input normalization, local redaction, tokenization or feature construction, representation/model inference, classifier decoding, and output serialization. All methods were measured on the same CPU and software environment with identical warm-up, repetition, threading, record order, and batch policy. We report P50/P95 latency and throughput for every baseline in Table [TIMING_TABLE].
```

### 放置与闭环

旧的 model-level latency 另表或删除；不得用 0.0078 ms frozen E5 与 4.56 ms full RC pipeline 直接排名。完成标准：timing raw logs 可审计。

## R3-14 Hospital use claim

### 审稿意见

> “The current data are public and de-identified, while the intended use includes hospital complaints and internal feedback. Please keep this limitation clear and avoid implying that the model has already been validated on hospital owned records.”

### 当前原文与位置

Markdown 第 43 行的原文为：

```text
Such feedback is collected through both public platforms and hospital complaint, survey, hotline, and follow-up systems.
```

Markdown 第 486 行的原文包含：

```text
No restricted patient-level clinical records were used in this study.
```

第 456 行虽写 `Hospital-owned complaint systems may differ...`，但摘要和结论仍使用容易被理解为已在 hospital-owned records 验证的广义 `hospital feedback` 表述。

### 修改细节与英文文字

执行 M10/M11。在摘要末和 `## 5.3` 增加：

```text
The evaluation uses public and de-identified proxy records rather than hospital-owned complaints, surveys, hotline records, or follow-up notes. Accordingly, the study demonstrates methodological feasibility on reproducible public data; it does not establish clinical, institutional, or operational validity on restricted hospital records.
```

### 放置与闭环

Data Availability、Limitations、Conclusion 三处保持同一措辞。若将来有真实医院数据，另写 external validation，不在本轮暗示已有。

## R3-15 Ethics and data collection

### 审稿意见

> “Please add a short statement explaining the ethics or exemption status for collecting the Chinese public web records and how the records were de-identified.”

### 当前原文与位置

Markdown 第 486 行的原文为：

```text
The remaining 1,049 records were sampled from publicly accessible Chinese web content collected by the authors using monitoring software. The analytic workflow excluded profile URLs, source URLs, and account-level metadata and applied identifier masking before modeling. No restricted patient-level clinical records were used in this study.
```

该段没有 ethics review/exemption 状态、collection authorization 或责任人说明。

### 修改细节与英文文字

必须先由用户确认真实伦理状态。不得自行写“IRB exempt”。根据实际情况选择以下唯一适用版本：

**若有正式审批：**

```text
The collection and analysis protocol was reviewed by [COMMITTEE_NAME] and approved/exempted under [APPROVAL_OR_EXEMPTION_ID] on [DATE]. The study used publicly accessible content, excluded profile URLs, source URLs, and account-level metadata, and applied [DE_IDENTIFICATION_PROCEDURE] before analysis. No restricted patient-level clinical records were accessed.
```

**若机构书面确认无需审查：**

```text
The authors obtained a written determination from [AUTHORIZED_BODY] that this analysis of publicly accessible, non-interventional data did not require [REVIEW_TYPE] ([DETERMINATION_ID], [DATE]). The collection excluded [FIELDS], and [DE_IDENTIFICATION_PROCEDURE] was applied before modeling. No restricted patient-level clinical records were accessed.
```

**若尚未获得确认：**

```text
The authors are obtaining an institutional determination concerning the ethics status of the public-web data collection. Until that determination is documented, the manuscript does not claim an ethics exemption and describes the data as publicly accessible and de-identified only.
```

### 修改后放置位置和闭环

在 `## Data Availability Statement` 之后新增 `## Ethics and Data Collection Statement`，并在 rebuttal 中说明附件/编号。必须用户提供 approval/exemption status、collection dates、de-identification SOP；没有这些信息不能提交最终版。

## R3-16 Reproducibility

### 审稿意见

> “Please provide the labeling prompts, redaction rules, complete training settings, and code or configuration used to reproduce the reported results.”

### 当前原文与位置

Markdown 第 126 行没有完整 prompts；第 204–214 行的规则原文只说明：

```text
The current implementation detects existing placeholders such as [PERSON] and [LOCATION], together with email addresses, URLs, Chinese mobile numbers, Chinese identification numbers, generic telephone numbers, and WeChat/QQ-style contact handles.
```

第 244 行只有主要模型超参数（`10,000 input features`、`384-dimensional shared layer`、`160 full-batch epochs`、`Adam`、`learning rate of 0.006`、`weight decay of 10−4`、`random seed 20260715`）；第 486 行给 GitHub archive，但没有 release manifest、运行命令、环境锁定和完整日志。

### 修改细节与英文文字

执行 M1/M3/M6/M8/M9/M12。在 `## Data Availability Statement` 末新增：

```text
The reproducibility archive contains the complete labeling prompts and JSON schema, redaction rules and regular expressions, category definitions, anonymized split manifest, training and validation configuration, random seeds, baseline search spaces, timing protocol, evaluation scripts, and environment lock file. The release is [REPOSITORY_URL] at tag [TAG_OR_COMMIT], with checksum [CHECKSUM]. Restricted, personal, account-level, and non-redistributable source content is excluded; the archive provides only permitted de-identified examples and record-level hashes needed to audit the split.
```

### 修改后放置位置和闭环

正文 Data Availability + Supplementary A–E + GitHub release。必须先实际打包并运行 clean-environment reproduction smoke test：从 manifest/config 开始至少能重建 split、训练一个 seed、计算指标并生成 timing schema。完成标准是他人不依赖作者口头解释即可复现主要表格，或明确列出不可公开部分及替代审计材料。

## 5. 原稿具体修改地图

| 原位置 | 原问题 | 修改动作 | 新位置 |
|---|---|---|---|
| Abstract，第 37 行 | 分类/医院应用表述偏强 | 改为 comparable classification + primary consistency claim；加入 public/de-identified proxy 限定 | Abstract |
| Introduction，第 43–47 行 | 贡献未突出 independent audit/source split | 重写贡献 1–3，加入 audit、group-aware split、changed consistency | Introduction |
| 2.2，第 65–67 行 | LLM 弱监督有说明但不够可复现 | 补 provenance、prompt sensitivity 与 audit 边界 | 2.2 |
| 3.2，第 124–142 行 | prompt、版本、裁决、schema 缺失 | 插入 M1/M2 段落，引用 Supplementary A/C | 3.2 |
| 3.4，第 204–214 行 | PII 规则边界未形成评估协议 | 明确 configured rules 与 unseen challenge 的区别 | 3.4 |
| 4.1，第 240–248 行 | split/source/seed/rare counts 不完整 | 加 leakage audit、language/source、audit sample、seed protocol | 4.1 |
| 4.2，第 287–315 行 | baseline tuning/timing/changed metric 不完整 | 加 hyperparameter table、fine-tuned encoder、changed denominator | 4.2 |
| 4.3，第 329–352 行 | 主贡献可能被理解为分类 superiority | 改为 comparable classification；加入 CI、语言/来源/rare 表 | 4.3 |
| 4.4，第 370–386 行 | overall consistency、PII 1.000 可能过度解释 | 加 changed-only 与 independent PII challenge | 4.4 |
| 4.5，第 398–420 行 | λ post-hoc、timing 不公平 | validation-only selection、多 seed、统一 end-to-end timing | 4.5 |
| Discussion，第 434–460 行 | 方法优势和医院用途边界需校准 | 重写 central contribution、deployment caveat、source/PII limitation | 5.1–5.3 |
| Conclusion，第 482 行 | 结果和适用范围偏强 | 用锁定 test、多 seed、audit 后数值；明确 proxy data | Conclusion |
| Data Availability，第 486 行 | ethics/reproducibility 缺失 | 加 ethics statement、release/tag/config | Data Availability 后 |

## 6. 实施顺序（必须按此顺序闭环）

### 阶段 A：先锁定事实，不能先润色

1. 备份投稿 Word 和 Markdown；建立 `revision-2026-10-08` 工作副本。
2. 收集 LLM 日志、prompt、settings、adjudication 记录、原始 source metadata、代码和环境文件。
3. 确认 ethics/data collection 状态。
4. 运行 M3 split audit；若发现 leakage，先重新划分并锁定新的 validation/test。

### 阶段 B：取得新增证据

5. 完成人审抽样和 agreement；形成 Supplementary C。
6. 完成 independent PII challenge set；形成 Supplementary D。
7. 仅用 validation 选择 λ、baseline hyperparameters 和 fine-tuned multilingual encoder。
8. 用多个 seed 在 locked test 上运行；导出 per-class、language/source、changed-only、CI。
9. 对所有重要 baseline 按统一边界计时。

### 阶段 C：写回正文和 rebuttal

10. 先更新 Tables/Supplementary，再写 Results，最后同步 Abstract/Introduction/Discussion/Conclusion。
11. 检查所有数字：数据量、support、macro-F1、consistency、CI、latency、memory、seed mean 必须在表、正文、摘要、结论中一致。
12. 为 R2-1 至 R2-5、R3-1 至 R3-16 各写一段“感谢—采取行动—新增位置—证据—限制”的 rebuttal。
13. 以投稿 Word 为最终排版源，导出 PDF 逐页检查表格、公式、图、参考文献和补充材料链接。

## 7. 最终提交前验收清单

- [ ] 真实 LLM provider/model/version/date/settings 已核对，`[PLACEHOLDER]` 全部清零。
- [ ] 两个完整 prompt、schema、retry 和 adjudication procedure 已归档。
- [ ] 独立人审已完成；human–LLM agreement 和 inter-rater agreement 有数值和 CI。
- [ ] exact/near duplicate、source group、synthetic lineage 跨 split 均有审计结论。
- [ ] Chinese/English/source-wise 结果和 support 已加入。
- [ ] medication 等 rare class 有 test support、per-class F1 和不确定性说明。
- [ ] λ 仅由 validation 选择；test 在选择后才打开；多 seed 已完成。
- [ ] PII independent challenge set、P/R/F1、FN examples 与 synthetic stress test 分开。
- [ ] changed/unchanged/all consistency 分母和 CI 正确。
- [ ] 所有 baseline 的 hyperparameters/search spaces 和 fine-tuned compact multilingual encoder 已报告。
- [ ] 所有重要方法在统一 end-to-end boundary 下计时。
- [ ] 摘要、引言、讨论、结论不再宣称非显著 classification superiority。
- [ ] public/de-identified proxy 与 hospital-owned validation 的边界清楚。
- [ ] ethics/data collection statement 使用真实 approval/exemption 状态。
- [ ] prompts/rules/config/code/split manifest/release tag 可复现或明确不可公开部分。
- [ ] Word 最终版与修订 Markdown 同步；文献、图表、公式和页内排版完成视觉检查。

## 8. 最终判断

**我可以独立完成的部分：** 全部文字重写、审稿意见矩阵、rebuttal、表格模板、统计/审计脚本、配置模板、版本同步和最终一致性检查。  
**我可以在你提供材料或允许本地运行后完成的部分：** split audit、重训、多 seed、language/source/rare-class、PII challenge 统计、agreement、CI、统一 latency 和所有结果回填。  
**必须由你或研究团队完成/确认的部分：** 真实人审、LLM 日志和 prompt 的真实性、裁决者信息、伦理状态、数据授权、未公开数据的处理和最终提交责任。  
**因此，本轮返修在方法和写作上可完成，但在证据层面尚未“自动完成”。** 在 M1–M9 的真实材料没有齐全前，不能诚实地把论文改成“已通过审稿要求”的最终稿；当前文件提供的是可以逐条执行、逐条验收的闭环实施方案。
