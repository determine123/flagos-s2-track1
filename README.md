# DeepSeek-V3 Decode GEMM：多后端 Triton 优化

面向 fused QKV-A 下投影的 tiny-M GEMM 工程实践：输入契约、独立参考实现、Triton kernel、跨后端适配和可追溯实验。

**2026-09-15 平台快照：8/8 芯片通过，平台平均加速比 2.50×，Task 66 实时榜第 8 名。**
数据由团队从 FlagOS 页面转录，不是本地重测或模型端到端加速。竞赛期间保持私有。

## 已验证结果

- 兼容性 5/8 → 8/8：为沐曦、燧原、海光提供固定配置版本。
- 燧原显示 0.00× → 0.22×：BLOCK_N/BLOCK_K 从32/32调整为64/64。
- 华为显示 0.01× → 1.82×：只改变B tile加载方向及转置表达，保持64/64配置。

见 [实验复盘](docs/EXPERIMENTS.md) 和 [平台结果转录](results/platform_submissions.csv)。

## 算子契约

```python
def dsv3_fused_a_gemm(mat_a, mat_b):
    # expected = (mat_a.float() @ mat_b.float()).to(mat_a.dtype)
    ...
    return out
```

A为[M,K] BF16行主序，M=1..16，K为256倍数；B为[K,N] BF16列主序视图，
stride=(1,K)，N为16倍数。输出[M,N] BF16，FP32累加。
7168×2112是代表权重形状，不限制全部合法输入。

## 快速开始

```bash
git clone https://github.com/determine123/flagos-s2-track1.git
cd flagos-s2-track1
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/check_environment.py
python -m pytest tests/ -q
python scripts/package_submission.py --output dist/task66-best.zip
```

无GPU时kernel用例明确跳过。厂商后端使用匹配的官方FlagTree环境，
普通kernel extra不保证八芯兼容。详细边界见 [复现说明](docs/REPRODUCIBILITY.md)。

### CUDA调用

```python
import torch
from dsv3_fused_a_gemm import dsv3_fused_a_gemm
a = torch.randn(4, 7168, dtype=torch.bfloat16, device="cuda")
w = torch.randn(2112, 7168, dtype=torch.bfloat16, device="cuda")
out = dsv3_fused_a_gemm(a, w.T)
```

### 本地性能测量

```bash
python benchmarks/bench_dsv3_fused_a_gemm.py --json local.bench.json
python benchmarks/bench_dsv3_fused_a_gemm.py --sweep --no-flush-l2
```

本地比较项目通用kernel与PyTorch BF16 matmul，不是官方评测程序；
不会自动加载厂商特化文件。CUDA-event耗时不等同完整Python请求墙钟时间。

## 实际目录

- src/dsv3_fused_a_gemm/：通用实现、公开包装层、独立FP32 PyTorch oracle。
- submissions/task66_perf_test_02/：平台2.50×最佳版本的五个源文件快照。
- submissions/task66/：较早的固定配置实验快照。
- tests/：CPU reference、CUDA kernel和提交结构检查。
- benchmarks/：CUDA计时与逐shape JSON。
- scripts/：环境报告、确定性打包和哈希清单。
- results/：平台转录数据，区分Failed与舍入后的0.00。
- integrations/vllm/：实验性接口适配，未验证整框架集成。
- docs/：规格、复盘、复现与面试材料。

项目reference是正确性标准；提交reference是调用Triton的入口，不能作为本地oracle。

## 完成度

- [x] 两参数契约、stride寻址、独立reference与测试框架
- [x] 平台八芯通过、后端实验隔离与最佳源码快照
- [x] 打包脚本、SHA-256与结果转录
- [ ] 目标设备逐case延迟和完整软件环境
- [ ] profiler / 编译产物解释性能改善原因
- [ ] 厂商特化的本地设备自动测试
- [ ] 端到端推理接入与收益验证

## 团队与来源

团队determine：李天明、xtty、5678_apix8yk、Wzp、8523_apiwmre。
初版由KernelGen Web生成，AI工具辅助审查、集成与实验；
个人贡献以实际PR和实验记录为准。本次为事后归档，不伪造历史开发时间。

参见 [协作规范](CONTRIBUTING.md)、[来源与许可](docs/PROVENANCE.md)、
[面试讲解](docs/INTERVIEW.md)。独立开源许可证尚待确认。
