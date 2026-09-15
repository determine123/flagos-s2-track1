# 复现说明

本地环境由 scripts/check_environment.py 输出。历史开发环境为Windows、PyTorch 2.8.0+cpu、Triton 3.4.0，无GPU。
平台完整设备型号、驱动和编译器版本未取得。普通CUDA测试不能证明厂商后端正确。

## 正确性

tests/test_reference.py验证独立参考实现；test_kernel.py在CUDA上验证项目通用kernel。
本地BF16容差rtol=0.02、atol=0.01是项目选择，未核对Task 66完整checker；不能称为官方容差。
平台特化快照具有历史平台结果，本地仅静态验证。CPU CI不能证明GPU正确性。

## Benchmark

CUDA Events计时，warmup/rep均为迭代次数，输出中位数毫秒。首次编译在warmup中触发。
输入生成不计时，输出分配调用在函数内，但事件耗时不等于完整Python请求延迟。
torch_contig的B预先转换，其数据不含转换成本。FP32 oracle与BF16性能对照分开。
可用256MiB缓冲区写入扰动缓存，不能保证所有硬件上冷缓存。保留逐shape原始JSON。
当前无本地GPU实测，历史平台成绩仅为转录。

## 打包

python scripts/package_submission.py --output dist/task66-best.zip

默认打包submissions/task66_perf_test_02五个源码；固定ZIP元数据，旁边生成manifest。
ZIP哈希可能与历史打包工具不同，但源码必须相同。脚本不证明设备执行或反作弊合规。
