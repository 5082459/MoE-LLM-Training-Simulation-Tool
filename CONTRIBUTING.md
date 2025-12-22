# 贡献指南

感谢你对 MoE-LLM 预训练资源需求分析工具的关注！请先阅读本指南，以便更高效地协作。

## 开发环境
- Python 3.9+
- `pip install -r requirements.txt`（建议使用 `python -m venv venv` 创建虚拟环境）
- 推荐在提交前同时运行后端与前端做一次冒烟检查：
  - 后端：`uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4`
  - 前端：`streamlit run frontend_app.py`

## 提交流程
1. Fork 仓库并创建功能分支（如 `feat/...`、`fix/...`）。
2. 开发与自测：确认前后端可以正常启动；如调整 GPU 画像/并行策略，请覆盖常见配置用例。
3. 文档：必要时更新 `README.md` / `CHANGELOG.md`，描述新特性或行为变化。
4. 提交信息：使用简洁前缀（`feat: ...`、`fix: ...`、`chore: ...`）。
5. 发起 PR，说明动机、主要改动与验证方式；如有行为兼容性风险，请在 PR 描述里标注。

## 代码风格与约定
- 遵循 PEP 8，保持函数/变量命名清晰；避免无注释的魔法常量。
- 新增 GPU 画像时更新 `GPU_Profiler_fork4.py`，并确保前端可选项/别名同步。
- 新增并行或精度策略时更新对应配置（`Parallel_fork4.py`、`Precision_Policy_fork4.py`）。
- 当前仓库缺少自动化测试；如能添加小型验证脚本或示例配置，请附在 PR 描述中。

## 问题反馈
- 通过 Issues 提交：请提供操作系统、Python 版本、复现步骤、期望结果与实际结果。
- 如涉及性能/精度评估，请附上相关模型与训练配置，便于重现与确认。
