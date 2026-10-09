# Arena revision v2 package

这是 IJSEKE-D-26-00185 返修实验的全新工作包。旧的 `major_revision_run` 仅作本地审计副本，不再作为运行目录，也不会被覆盖。

## 上传内容

- `input/`：论文方法、数据集、编辑决定与审稿意见、逐条返修计划；
- `code_reference/`：现有参考实现和本地脱敏规则；
- `instructions/ARENA_RUN_PROMPT.txt`：上传整个工作包后，直接粘贴到新的 Arena 云环境；
- `outputs/`：预留给云环境返回的新结果，不需要作为输入上传。

## 操作顺序

1. 将 `input/`、`code_reference/` 和 `instructions/ARENA_RUN_PROMPT.txt` 上传到新的 Arena 任务。
2. 先执行提示词中的 PHASE 1 输入核对。
3. 核对通过后再让云环境继续完整运行。
4. 云端输出目录命名为 `major_revision_run_v2`，复制回本地后放入 `outputs/` 或同级新目录。
5. 本地先审查梯度检查、数据哈希、结果 CSV 和最终报告，再决定是否修改论文。

## 不上传内容

不要上传旧 checkpoint、旧 `fitted_models.pkl`、旧结果 CSV、旧预测文件或整个旧 `major_revision_run`。这些文件既占空间，也可能让新实验误用未经验证的结果。
