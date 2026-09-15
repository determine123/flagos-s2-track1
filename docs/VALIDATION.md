# 本次工程归档验证

日期：2026-09-15。本地Python 3.9.25，Windows，PyTorch 2.8.0 CPU环境。

- python -m pytest tests/ -q：90 passed，30 skipped。
- 29个CUDA相关用例因无GPU跳过，1个集成前stub用例跳过。
- 最佳五个源码SHA-256与历史perf_test_02报告一致。
- 确定性打包测试比较两次ZIP及manifest，CRC和载荷字节一致。
- 新打包ZIP不是历史提交ZIP；源文件一致，ZIP元数据标准化。
- git diff --check通过；凭据模式扫描未发现JWT/Bearer/GitHub Token。

GPU执行、厂商性能和GitHub CI状态不由这些本地检查证明。
平台成绩来源与本地验证分开记录。
