# DSV3 Fused-A GEMM Optimization with Triton

> FlagOS Open Computing Challenge S2 · 赛道一 · Task 66
>
> 一个面向 DeepSeek-V3 decode 场景的 **skinny-M GEMM** 工程：从算子规格分析、PyTorch 参考实现、Triton baseline、性能调优，到多后端验证、提交归档和最终工程化交付。

[![Status: Baseline](https://img.shields.io/badge/status-baseline-orange)](#当前状态)
[![Backend: SM90+](https://img.shields.io/badge/backend-SM90%2B-blue)](#环境与复现)
[![License: TBD](https://img.shields.io/badge/license-TBD-lightgrey)](#许可证与公开计划)

## 项目定位

本项目起源于 FlagOS S2 Task 66，但目标不止于提交一个 `submission.py`。我们希望完整展示一个高性能 GPU kernel 从规格理解到工程交付的全过程：

- 为 decode 小 batch 场景设计适合 `num_tokens <= 16` 的 Triton GEMM；
- 理解 BF16 输入、FP32 累加和列主序权重访问的性能与精度影响；
- 用可复现的测试和 benchmark 支撑每一次优化结论；
- 记录 baseline、实验、失败原因、平台结果及其对应的 Git commit；
- 最终沉淀为可讲解、可维护、可继续扩展的高性能算子项目。

竞赛成绩是外部评测证据，**工程完整性、技术积累和可复现性是第一目标**。README 和结果表只记录实际测得的数据，不臆测性能或兼容性结论。

## Task 66 规格

### 计算定义

```python
def reference(mat_a, mat_b):
    return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)
```

| 项目 | 约束 |
|---|---|
| `mat_a` | `[num_tokens, hd_in]`，BF16，row-major |
| `mat_b` | `[hd_in, hd_out]`，BF16，列主序；等价于 row-major `[hd_out, hd_in]` 权重的 `.t()` |
| `num_tokens` | `1 <= num_tokens <= 16` |
| `hd_in` | 256 的倍数 |
| `hd_out` | 16 的倍数 |
| 输出 | `[num_tokens, hd_out]`，dtype 与 `mat_a` 一致 |
| 累加 | FP32 |
| 目标环境 | SM90+；其他后端以实际验证结果为准 |

核心计算必须由 Triton/Triton-TLE 自定义 kernel 完成。不得使用 PyTorch 内置矩阵乘法替代核心路径。

## 当前状态

- **核心赛题**：Task 66 · `dsv3_fused_a_gemm`
- **项目阶段**：Baseline 开发
- **正确性**：待验证
- **当前最佳性能**：待测试
- **支持后端**：以实际验证结果为准
- **竞赛批次**：第 5 批
- **仓库可见性**：竞赛期间 Private

> 性能数据必须同时记录硬件、驱动、软件版本、shape、dtype、计时方法和对应 commit。没有这些信息的数据不作为项目结论。

## 目录规划

```text
flagos-s2-track1/
├── README.md
├── .gitignore
├── pyproject.toml                 # 可选：格式化、测试和静态检查
├── requirements.txt               # 可复现依赖（按官方环境填写）
├── src/
│   ├── reference.py               # PyTorch 参考语义
│   ├── baseline.py                # 最小正确 Triton 实现
│   ├── optimized.py               # 当前优化实现
│   └── configs.py                 # shape/后端配置
├── tests/
│   ├── test_correctness.py
│   ├── test_edge_cases.py
│   └── test_regression.py
├── benchmarks/
│   ├── benchmark.py
│   ├── shapes.py
│   └── results/                   # 只提交小型文本结果，不提交大文件
├── docs/
│   ├── architecture.md            # kernel 映射与内存访问设计
│   ├── optimization.md            # 优化实验与证据
│   ├── multi_backend.md           # 多芯片验证
│   ├── environment.md             # 环境说明
│   ├── decisions.md               # 技术决策记录
│   └── interview_notes.md         # 项目讲解材料
├── submissions/
│   ├── README.md                  # 平台提交总台账
│   └── task66/                    # 文件、结果与 commit 映射
└── scripts/
    ├── run_tests.py
    ├── run_benchmark.py
    └── package_submission.py
```

目录可以随项目演进；新增代码、测试和文档时优先保持职责清晰，避免把实验文件直接堆在仓库根目录。

## 团队

| 成员 | GitHub ID | 主要职责 | 备用职责 |
|---|---|---|---|
| 李天明 | `determine123` | 项目负责人、架构、版本集成、最终 PR | 故障接管 |
| xtty | 待填写 | Triton GEMM baseline、正确性 | Task 66 性能优化 |
| 5678_apix8yk | 待填写 | skinny-M 性能优化 | 参数实验 |
| Wzp | 待填写 | 测试矩阵、回归和结果整理 | 打包检查 |
| 8523_apiwmre | 待填写 | 参数搜索、多后端适配、性能分析 | 题目接管 |

每项任务指定一名最终负责人。协作者可以多人，但负责人对正确性、版本和结果归档负责。

## 环境与复现

以比赛官方环境为准；本地环境用于提前发现问题，不能替代平台结果。请在 `docs/environment.md` 记录：Python、PyTorch、Triton/Triton-TLE、驱动、设备、操作系统和后端版本。

```bash
git clone https://github.com/determine123/flagos-s2-track1.git
cd flagos-s2-track1
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

环境检查：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import triton; print(triton.__version__)"
```

计划中的标准入口：

```bash
python scripts/run_tests.py
python scripts/run_benchmark.py
python scripts/package_submission.py
```

在这些脚本完成前，请以各模块中的实际命令为准，不要把“计划入口”误认为已经实现的功能。

## 技术路线

1. 明确接口、shape、stride、dtype 和容差，建立 PyTorch reference。
2. 实现最小、可解释、可验证的 Triton GEMM baseline。
3. 针对 `M=num_tokens <= 16` 研究 skinny-M 的 program 映射与 launch 开销。
4. 分析 `BLOCK_M/BLOCK_N/BLOCK_K`、`num_warps`、`num_stages` 和配置矩阵。
5. 研究列主序 `mat_b` 的加载、合并访存及 FP32 累加策略。
6. 每次只改变一个主要变量，保存修改前后的正确性和性能证据。
7. 在所有公开 shape、边界用例和已支持后端上回归。
8. 将平台候选版本与唯一 Git commit、文件 SHA256 和结果归档对应起来。

## 正确性规范

至少覆盖：最小输入、常规输入、非 2 的幂、尾块、`num_tokens=1/16`、全零、正负混合、极值、随机值、重复值以及题目要求的 dtype。

- Exact 题使用 `torch.equal`；
- 容差题使用题目规定的 `atol/rtol`；
- 特殊 mismatch 比例必须按官方公式实现；
- 随机测试固定种子，并记录 shape、dtype、参数、误差位置和 commit；
- 测试失败不得被静默吞掉。

参考对比：

```python
expected = (mat_a.float() @ mat_b.float()).to(mat_a.dtype)
actual = candidate(mat_a, mat_b)
torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
```

## 性能测试规范

- warmup 后再计时；
- 不把输入生成、D2H 同步和结果校验计入 kernel 时间；
- 同一组数据使用一致的计时方法；
- 记录中位数、尾延迟、shape、dtype、设备、软件版本和 commit；
- 小 `M` 重点观察 launch-bound；GEMM 重点观察访存、计算和 tile 利用率；
- 优化没有稳定复现或破坏其他 shape 时，不作为改进结论。

### 性能结果模板

| Case | Shape `(M,K,N)` | Backend | Baseline | Current | Speedup | Commit | 备注 |
|---|---:|---|---:|---:|---:|---|---|
| Case 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 多后端验证

| 后端/芯片 | 编译 | 正确性 | 性能 | Commit | 失败原因/备注 |
|---|---|---|---|---|---|
| 天数 | 待测 | 待测 | 待测 | 待测 | |
| 沐曦 | 待测 | 待测 | 待测 | 待测 | |
| 燧原 | 待测 | 待测 | 待测 | 待测 | |
| 海光 | 待测 | 待测 | 待测 | 待测 | |
| 昆仑芯 | 待测 | 待测 | 待测 | 待测 | |
| 华为 | 待测 | 待测 | 待测 | 待测 | |
| 国际通用芯片 A | 待测 | 待测 | 待测 | 待测 | |
| 国际通用芯片 B | 待测 | 待测 | 待测 | 待测 | |

状态统一使用：`待测`、`编译失败`、`运行失败`、`正确性失败`、`性能未过线`、`有效`、`当前最佳`。未实际验证的后端不得标记为支持。

## 协作流程

### 分支

- `main`：稳定、可回退，只通过 PR 合并；
- `feat/task66-xxx`：新功能；
- `perf/task66-xxx`：性能优化；
- `fix/task66-xxx`：问题修复；
- `test/task66-xxx`：测试；
- `docs/xxx`：文档。

```bash
git checkout main
git pull origin main
git checkout -b perf/task66-skinny-m
git add .
git commit -m "perf(task66): tune skinny-m tile"
git push -u origin perf/task66-skinny-m
```

随后创建 Pull Request，至少一名非作者成员 Review 通过后再合并。禁止直接推送 `main`、覆盖公共历史或未经测试提交平台。

### Commit 格式

```text
<type>(<scope>): <description>
```

类型包括 `feat`、`fix`、`perf`、`test`、`docs`、`refactor`、`chore` 和 `revert`。示例：

```text
feat(task66): add correct triton baseline
fix(task66): handle non power of two output width
perf(task66): tune block size for decode shapes
test(task66): add bf16 edge cases
```

## Pull Request 清单

- [ ] 接口签名、输出 dtype 和布局符合题目要求
- [ ] 正确性测试通过，包含边界用例
- [ ] 实际执行路径运行 Triton kernel
- [ ] 无 PyTorch fallback、设备规避分支或隐藏异常
- [ ] 已记录 benchmark 环境和可复现命令
- [ ] 性能变化有数据证据，且未造成回归
- [ ] 无凭据、绝对路径、缓存或大文件
- [ ] 已指定对应 Git commit
- [ ] 至少一名队友完成 Review

## 反作弊与代码边界

所有参赛实现必须遵守官方规则：核心计算完全基于 Triton 或 Triton-TLE，禁止用 PyTorch 内置算子替代核心计算；禁止通过 `try/except`、条件分支、设备判断或其他方式在 Triton 失败时 fallback 到 Torch；禁止硬编码隐藏测试输出。发现不确定行为时先暂停提交并核对官方规则。

KernelGen 可以辅助生成 baseline、提出参数组合和分析错误，但生成代码必须由团队成员人工阅读、解释和测试后才能进入 PR。需要能说明 grid、program 映射、mask、指针偏移、dtype 转换和累加精度。

## 平台提交与版本归档

每次提交至少保存：

```text
submissions/task66/YYYYMMDD_HHMM_vNNN/
├── submission.py
├── metadata.md
└── result.md
```

`metadata.md` 记录赛题、时间、提交人、分支、Git commit、文件 SHA256、本次假设和与上一版的差异；`result.md` 记录正确性、加速比、排名、错误日志、是否成为最佳和下一步。

建议在每个重要版本打标签：

```bash
git tag task66-correct-v1
git tag task66-platform-best-v1
git push origin --tags
sha256sum submission.py  # Windows 可使用 Get-FileHash
```

提交额度按官方规则和队内台账管理。未登记 commit、未通过本地检查或连续重复失败的版本不得盲目提交。

## 保密与公开计划

竞赛期间仓库保持 Private。禁止提交账号密码、Token、Cookie、SSH 私钥、`.env`、未公开测试数据、平台敏感链接和未经授权的第三方代码。若凭据误提交，应立即撤销/轮换并检查 Git 历史。

待官方规则允许公开后，再清理账号信息、平台专有文件、不可公开数据和内部记录，补充许可证、公开环境说明和可运行示例，将本项目整理为公开作品集。

## 项目完成标准

- 新环境可以根据文档独立复现；
- baseline 与 optimized 均有清晰实现和对比；
- 公开 shape 和边界测试自动化通过；
- 每轮优化都有数据、假设和失败记录；
- 有性能回归和版本归档工具；
- 实际支持的后端和性能结论均有证据；
- 每位成员都能说明自己的贡献；
- 能形成技术文章和 5–20 分钟项目演示；
- 平台最佳版本与最终交付版本可以通过 commit 和 SHA256 一一对应。

## 当前行动清单

- [ ] 确认全部成员已接受私有仓库邀请
- [ ] 完善成员 GitHub ID 和职责
- [ ] 建立项目目录及环境记录
- [ ] 写出 PyTorch reference 和 Triton baseline
- [ ] 建立正确性测试矩阵
- [ ] 建立 benchmark 与结果归档
- [ ] 完成第一版 Task 66 baseline
- [ ] 记录第一次可复现性能结果
- [ ] 建立 PR、Issue 和技术决策流程
- [ ] 规划公开作品集清理和发布

## 许可证

竞赛期间暂不添加许可证。公开前请根据官方仓库要求、第三方依赖许可证和团队贡献约定补充适当许可证。

## 联系与记录

技术问题写入 GitHub Issues，代码讨论写入 Pull Request，重要结论写入 `docs/decisions.md`。群聊用于快速沟通，但不能作为唯一记录。

---

本仓库仅用于团队学习、工程协作和参加 FlagOS Open Computing Challenge S2。所有代码、测试和提交必须遵守赛事规则、官方贡献指南及相关开源项目许可证。
