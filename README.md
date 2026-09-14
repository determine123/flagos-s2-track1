# DSV3 Fused-A GEMM Optimization with Triton

面向 DeepSeek-V3 Decode 场景的 Skinny-M GEMM 实现、性能优化与工程化实践。

本项目源于 FlagOS Open Computing Challenge S2 赛道一 Task 66。团队目标不是仅完成一个比赛提交文件，而是围绕真实的大模型推理算子，完整实践算子规格分析、PyTorch Reference、Triton Kernel、正确性验证、GPU 性能分析、Skinny-M 调优、多形状与多芯片测试、自动化 Benchmark、实验版本追踪、工程文档和就业作品集建设。

> 竞赛成绩是外部评测证据，但不是项目是否成功的唯一标准。所有性能和兼容性结论均以实际测试为准。

## 1. 项目状态

- 项目名称：DSV3 Fused-A GEMM Optimization
- 对应赛题：Task 66 · `dsv3_fused_a_gemm`
- 所属路径：`gemm/dsv3_fused_a_gemm`
- 团队名称：`determine`
- 当前批次：第 5 批
- 当前公开提交量：30 次
- 比赛截止：2026 年 9 月 17 日 19:59（北京时间）
- 当前阶段：Baseline 设计与环境验证
- 当前最佳加速比：待实测
- 当前平台排名：待实测
- 仓库状态：竞赛期间保持 Private

### 项目完成度

- [ ] 规格与边界条件整理完成
- [ ] PyTorch Reference 可独立运行
- [ ] Triton Baseline 正确性通过
- [ ] 建立自动化正确性测试
- [ ] 建立统一 Benchmark
- [ ] 完成第一轮参数搜索
- [ ] 完成 Skinny-M 专项优化
- [ ] 完成主要形状性能回归
- [ ] 完成平台多芯片验证
- [ ] 完成技术报告、架构图和结果图
- [ ] 完成 5～10 分钟项目演示
- [ ] 完成可公开作品集版本

## 2. 项目背景

DeepSeek-V3 的注意力预处理包含融合后的 QKV-A 下投影：

```text
out = mat_a @ mat_b
```

在 Decode 场景中，输入 token 数很少（`num_tokens <= 16`），而权重矩阵规模较大，M 维非常窄，权重读取和 Kernel launch 可能成为主要成本。因此，本项目使用手写 Triton Kernel，针对 Skinny-M 场景研究小批量、低延迟 GEMM 的映射、访存、数据复用和参数调优。

## 3. 任务规格

### 接口与计算

最终提交函数的参数数量、顺序和调用语义必须与题目要求完全一致：

```python
def reference(mat_a, mat_b):
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)
```

| 项目 | 约束 |
|---|---|
| `mat_a` | `[num_tokens, hd_in]`，BF16，row-major |
| `mat_b` | `[hd_in, hd_out]`，BF16，列主序转置视图 |
| `num_tokens` | `1 <= num_tokens <= 16` |
| `hd_in` | 256 的倍数 |
| `hd_out` | 16 的倍数 |
| 输出 | `[num_tokens, hd_out]`，dtype 与 `mat_a` 一致 |
| 累加 | FP32 |
| 目标环境 | SM90+；其他后端以实际验证为准 |

`mat_b` 可理解为：

```python
weight = torch.empty(hd_out, hd_in, dtype=torch.bfloat16, device="cuda")
mat_b = weight.t()
```

实现不能只根据表面 shape 假设物理布局，必须正确处理实际 stride。正确性采用官方规定的 per-dtype tolerance，本地不得自行放宽。

## 4. 项目目标

### 正确性

- 签名、shape、dtype、布局和 FP32 累加语义完全一致；
- 正确读取列主序 `mat_b`；
- 覆盖 `M = 1～16`、多组 K/N 和边界 shape；
- 不发生越界访问或未初始化写回；
- 所有正式测试可复现。

### 性能

- 建立稳定、可解释的 Triton Baseline；
- 降低 Decode 小 batch 的 Kernel 延迟；
- 研究 `BLOCK_M/BLOCK_N/BLOCK_K`、`num_warps`、`num_stages`、权重复用和 M 分组；
- 建立性能回归机制，并为每项优化提供数据证据。

### 工程与就业展示

- 新成员可按 README 复现；
- 测试、Benchmark、提交代码相互独立；
- 每次平台提交可映射到唯一 Git commit；
- 保留成功和失败实验；
- 能解释 Skinny-M、布局、tile、瓶颈和调优依据；
- 最终形成技术文章、Demo 和面试材料。

## 5. 仓库结构

```text
flagos-s2-track1/
├─ README.md
├─ LICENSE                         # 公开前根据规则补充
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ src/dsv3_fused_a_gemm/
│  ├─ __init__.py
│  ├─ reference.py
│  ├─ baseline.py
│  ├─ optimized.py
│  ├─ kernel.py
│  ├─ configs.py
│  └─ operator.py
├─ tests/
│  ├─ conftest.py
│  ├─ test_correctness.py
│  ├─ test_shapes.py
│  ├─ test_layout.py
│  ├─ test_numerics.py
│  └─ test_regression.py
├─ benchmarks/
│  ├─ benchmark.py
│  ├─ benchmark_reference.py
│  ├─ benchmark_kernel.py
│  ├─ shapes.py
│  ├─ metrics.py
│  └─ results/
├─ scripts/
│  ├─ check_environment.py
│  ├─ run_correctness.py
│  ├─ run_benchmark.py
│  ├─ run_autotune.py
│  ├─ compare_results.py
│  └─ package_submission.py
├─ submissions/task66/YYYYMMDD_HHMM_v001/
│  ├─ submission.py
│  ├─ metadata.md
│  └─ result.md
└─ docs/
   ├─ specification.md
   ├─ architecture.md
   ├─ correctness.md
   ├─ benchmark_methodology.md
   ├─ optimization_log.md
   ├─ multi_backend.md
   ├─ decisions.md
   ├─ final_report.md
   └─ interview_guide.md
```

## 6. 快速开始

```bash
git clone https://github.com/determine123/flagos-s2-track1.git
cd flagos-s2-track1
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

以官方环境为准，并在 `docs/environment.md` 记录 Python、PyTorch、Triton/Triton-TLE、驱动、设备、操作系统和后端版本。

```bash
python scripts/check_environment.py
python scripts/run_correctness.py
python scripts/run_benchmark.py
python scripts/run_autotune.py
# 或 pytest tests -v
```

上述脚本尚未实现前，以各模块中的实际命令为准；计划入口不代表功能已经完成。

## 7. Kernel 设计与优化路线

第一版 Baseline 优先保证正确性：一个 program 处理 M/N tile，沿 K 循环加载，使用 `tl.dot` 和 FP32 accumulator，按实际 stride 读取 `mat_b`，并对 M/N 尾块使用 mask。设计示意：

```python
pid_n = tl.program_id(0)
offs_m = tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = load_a(offs_m, k)
    b = load_b(k, offs_n)       # 使用真实 stride
    acc += tl.dot(a, b)
store_output(acc)
```

后续研究：

- `BLOCK_M = 1/2/4/8/16`，让权重 tile 服务多行输入；
- `BLOCK_N = 16/32/64/128/256`，平衡并行度、寄存器和写回；
- `BLOCK_K = 32/64/128/256`，观察 dot 利用率和 pipeline；
- `num_warps = 2/4/8`、`num_stages = 2/3/4/5`；
- 按 `M=1`、`2`、`3～4`、`5～8`、`9～16` 分组；
- 权重复用、launch 开销、寄存器压力和不同后端差异。

每次只改变一个主要变量，并记录假设、环境、commit、正确性、所有 shape 的性能变化及最终结论。

## 8. 正确性测试

至少覆盖：`M=1,2,3,4,8,15,16`；K 为 256 倍数；N 为 16 倍数；最小和较大 K/N；N 为或不为 `BLOCK_N` 整数倍；全零、正负混合、极值、随机值、重复值和不同随机种子。

必须测试真实转置视图：

```python
weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
mat_b = weight.t()
```

示例：

```python
expected = reference(mat_a, mat_b)
actual = optimized(mat_a, mat_b)
torch.testing.assert_close(actual, expected, atol=OFFICIAL_ATOL, rtol=OFFICIAL_RTOL)
```

失败时记录：seed、M/K/N、dtype、stride、最大绝对/相对误差、mismatch、错误位置、Kernel 配置、设备、编译器版本和 Git commit。

## 9. Benchmark 规范

先 warmup 再正式计时；不将输入生成、D2H 同步和正确性检查计入 kernel 时间；不在计时区打印；Baseline 与 Triton 使用一致输入和计时方式；记录中位数、P95、波动、硬件、软件版本、shape、dtype、配置和 commit。

```text
speedup = baseline_latency / triton_latency
FLOPs = 2 × M × K × N
```

| M | K | N | Baseline/μs | Triton/μs | Speedup | BLOCK_M | BLOCK_N | BLOCK_K | Warps | Stages |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 待填写 | 待填写 |  |  |  |  |  |  |  |  |
| 2 | 待填写 | 待填写 |  |  |  |  |  |  |  |  |
| 4 | 待填写 | 待填写 |  |  |  |  |  |  |  |  |
| 8 | 待填写 | 待填写 |  |  |  |  |  |  |  |  |
| 16 | 待填写 | 待填写 |  |  |  |  |  |  |  |  |

## 10. 多后端验证

| 芯片 | 编译 | 正确性 | 性能 | 加速比 | 状态 | 备注 |
|---|---|---|---|---:|---|---|
| 天数 | 待测 | 待测 | 待测 |  | 待测 | |
| 沐曦 | 待测 | 待测 | 待测 |  | 待测 | |
| 燧原 | 待测 | 待测 | 待测 |  | 待测 | |
| 海光 | 待测 | 待测 | 待测 |  | 待测 | |
| 昆仑芯 | 待测 | 待测 | 待测 |  | 待测 | |
| 华为 | 待测 | 待测 | 待测 |  | 待测 | |
| 国际通用芯片 A | 待测 | 待测 | 待测 |  | 待测 | |
| 国际通用芯片 B | 待测 | 待测 | 待测 |  | 待测 | |

状态只使用：`待测`、`编译失败`、`运行失败`、`正确性失败`、`性能未过线`、`有效`、`当前最佳`。不把单一 GPU 的成功描述为全部后端支持。

## 11. KernelGen 与反作弊

KernelGen 可用于生成 Baseline、分析编译错误、提出参数组合、补充测试和辅助性能分析，但生成代码必须由成员人工阅读、解释和测试。验收必须能说明 grid、program 映射、tile、stride、mask、累加精度和执行路径。

核心计算必须完全基于 Triton 或 Triton-TLE。禁止：

- 用 PyTorch 内置矩阵乘法替代提交路径；
- Triton 失败后 fallback 到 Torch；
- 用 `try/except` 隐藏编译或运行失败；
- 根据设备或芯片绕过 Triton；
- 硬编码隐藏测试输出；
- 实际执行路径不运行自定义 Kernel。

PyTorch 仅用于 Reference、测试数据、正确性对比和本地 Baseline Benchmark。

## 12. Git 协作规范

`main` 只保存稳定、可回退且经过 Review 的版本，禁止直接推送和 force push。分支命名：

```text
feat/task66-baseline
perf/task66-skinny-m
perf/task66-weight-reuse
fix/task66-layout
test/task66-correctness
bench/task66-shapes
docs/task66-report
```

```bash
git checkout main
git pull origin main
git checkout -b perf/task66-skinny-m
git add .
git commit -m "perf(task66): tune skinny-m tile"
git push -u origin perf/task66-skinny-m
```

Commit 格式：`<type>(task66): <description>`。类型包括 `feat`、`fix`、`perf`、`test`、`bench`、`docs`、`refactor`、`chore`、`revert`。每个 PR 至少一名非作者成员 Review，必须包含修改目的、方案、测试、性能、环境、风险、回退方案和实验编号。

## 13. 团队成员与职责

| 成员 | GitHub ID | 主要职责 | 就业成果 |
|---|---|---|---|
| 李天明 | `determine123` | 项目管理、架构、集成、提交管理 | 项目架构与完整交付 |
| xtty | 待填写 | Triton Baseline、正确性 | Kernel 基础实现 |
| 5678_apix8yk | 待填写 | Skinny-M 优化 | GEMM 性能优化 |
| Wzp | 待填写 | 测试、回归、结果整理 | 测试与质量体系 |
| 8523_apiwmre | 待填写 | 参数搜索、多芯分析 | 性能分析与自动调优 |

每位成员保留自己完成的 Issue、Commit、PR、测试/文档和技术结论。

## 14. 平台提交管理

每日额度按官方规则和团队台账集中管理。未登记 commit、未通过本地检查或连续重复失败的版本不得提交。

```text
submissions/task66/YYYYMMDD_HHMM_vNNN/
├─ submission.py
├─ metadata.md
└─ result.md
```

每次归档至少记录提交人、时间、分支、Git commit、文件 SHA256、实验假设、Kernel 配置、正确性、平均加速比、排名、失败原因和下一步。重要版本可打标签：

```bash
git tag task66-correct-v1
git tag task66-platform-best-v1
git push origin --tags
# Linux: sha256sum submission.py
# Windows PowerShell: Get-FileHash submission.py -Algorithm SHA256
```

## 15. 项目路线图

1. **正确性 Baseline**：环境、规格、Reference、KernelGen 方案、手写 Triton Baseline。
2. **统一 Benchmark**：标准 Shape、固定计时、环境记录、CSV/JSON 结果。
3. **性能优化**：tile、warp/stage、M 分组、权重复用、寄存器压力分析。
4. **平台验证**：提交闭环、多芯片结果、最佳版本映射。
5. **项目化整理**：架构图、性能图、技术报告和演示。
6. **公开作品集**：规则允许后清理账号、平台专有文件和不可公开数据，补充许可证和复现说明。

## 16. 就业作品集交付物

最终形成：清晰的 GitHub 仓库、技术文章《面向 DeepSeek-V3 Decode 的 Skinny-M GEMM Triton 优化实践》、5～10 分钟演示、性能图表、失败实验记录和每名成员可独立讲解的贡献说明。

简历描述模板：

```text
参与 DeepSeek-V3 Decode 场景 Fused QKV-A Skinny-M GEMM 优化项目，
基于 Triton 实现 BF16 输入、FP32 累加的自定义矩阵乘 Kernel，负责<个人职责>，
建立覆盖多 Shape、转置权重布局和数值误差的正确性测试及性能 Benchmark，
并通过参数搜索、权重复用和执行配置调优分析小 Batch 推理场景的延迟瓶颈。
```

实测结果完成后再填写具体硬件、Baseline、加速比和百分比，禁止使用无证据数字。

## 17. 安全、保密与公开计划

竞赛期间仓库保持 Private。禁止提交密码、Token、Cookie、SSH 私钥、`.env`、私人联系方式、未公开测试数据、对手代码、来源不明实现、大型缓存和平台敏感文件。凭据误提交时立即撤销/轮换，并检查 Git 历史。

比赛规则允许公开后，再清除账号信息、不可公开数据和平台专有文件，检查第三方许可证，补充 License 和公开环境说明。竞赛期间暂不添加独立许可证。

## 18. 当前行动清单

- [ ] 确认全部成员接受邀请
- [ ] 完善成员 GitHub ID 和职责
- [ ] 建立项目目录和环境记录
- [ ] 完成 PyTorch Reference 与 Triton Baseline
- [ ] 建立正确性测试矩阵和 Benchmark
- [ ] 完成第一轮参数搜索
- [ ] 建立 PR、Issue 和技术决策流程
- [ ] 完成第一版 Task 66 baseline
- [ ] 记录可复现性能结果
- [ ] 规划公开作品集清理和发布

## 19. 联系与声明

项目负责人：李天明；团队：`determine`。代码讨论使用 Pull Request，任务跟踪使用 GitHub Issues，决策记录在 `docs/decisions.md`，实验记录在 `docs/optimization_log.md`，平台记录在 `submissions/`。重要技术结论不能只保存在群聊中。

本项目用于学习和研究大模型推理场景中的 GPU Kernel 开发与性能优化。性能结果仅对明确记录的硬件、软件版本、输入 Shape 和测试方法负责。所有代码、测试和提交必须遵守赛事规则、官方贡献指南及相关开源项目许可证。

## 20. 致谢

感谢 FlagOS Community、Triton、PyTorch、DeepSeek、KernelGen 和 FlagTree 提供的技术基础与评测环境。具体引用和许可证信息将在项目公开前补充完整。
