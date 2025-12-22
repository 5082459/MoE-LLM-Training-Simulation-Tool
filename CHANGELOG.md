# Changelog

## [v1.0.0] - 2025-12-22
- 初始开源发布：MoE-LLM 预训练资源需求分析工具。
- Tab 1：模型参数/激活/显存容量估算，单位 step FLOPs 与吞吐评估，提供 GPU 选型建议。
- Tab 2：支持 DP/PP/TP/EP 组合的通信量与时间估算，区分 scale-up / scale-out。
- 时间线推演：step 耗时、GPU Hours、训练天数计算。
- 提供一键启动脚本（`start.sh` / `start.bat`）与 Streamlit 可视化前端。
