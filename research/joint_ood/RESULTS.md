# 阶段三实验结果汇总：联合动作 OOD 与模型利用

研究问题：在基于模型的离线 MARL 中，当每个智能体的动作在数据里都出现过、但它们的组合没出现过时（marginal-in, joint-out），学到的模型是否会"自信地错"？基于集成分歧（ensemble std）的不确定性惩罚能否发现这种错误？

预期假设：
- H1：模型在 joint-OOD 动作上系统性乐观，而集成 std 远小于真实误差（低估）。
- H2：因此 ensemble 惩罚压不住利用；只有显式的联合支撑（joint support）约束才有效。
- H3：失败的根源是联合覆盖不足，而不是其他因素（数据量、容量、horizon 等）。
- H4：问题随智能体数 n 增大而加重。

所有数字都是均值 ± 95% CI（t 区间），按 seed 配对检验。原始数据和表格都在 `research/joint_ood/` 下，所有结果已推送到分支 `claude/project-thread-j063fs`。

---

## 1. Pilot 1：单步分配博弈

n 个智能体分配到 n 个地标，数据只包含排列（没有两个智能体去同一地标）。比较三种奖励模型：joint MLP、per-agent attention、per-agent additive。5 成员集成，然后用梯度规划器找最优联合动作。

文件：`results_4080.json` + `results_4080_seeds5-9.json`（主实验，10 seeds），`ablations/`（16 个消融，n=4,8 × 5 seeds），`ablation_summary.txt`，`main_10seeds_ci.txt`。

### 1.1 主结果（10 seeds）
- 在"边缘见过、组合没见过"的动作上，ensemble std 能把它和分布内动作分开（AUROC 0.67–1.0），但 std/err 只有 0.04–0.5，也就是 **std 比真实误差小一个数量级**。→ 支持 H1。
- n=8 时，joint 和 additive 在 ensemble 惩罚下规划出的动作几乎全是重复分配（dup 0.8–1.0），exploit gap 约 0.2–0.3。→ 支持 H2、H4。
- 联合支撑惩罚（plus_joint_support_oracle）在所有设置下 dup = 0、gap ≈ 0。→ 支持 H2。

### 1.2 消融（16 个，每次只改一个设置）
- 改变 ensemble 大小 K（2/10/20）、数据量（2k/100k）、宽度（64/1024）、训练步数（1k/10k）、bootstrap 开关、噪声（0.05/0.2）、规划器 restarts 和 init，**n=8 的失败都还在**。→ 支持 H3。
- λ 加大到 300 也没用：joint n=8 的 dup 仍有 0.6–1.0，additive n=8 的 regret 反而从 0.25 升到 0.34。
- K 变大能提高 AUROC（joint n=8 从 0.89 到 0.96），但规划结果没变好。说明问题出在**校准**，不出在排序。
- 规划器只从 logged 联合动作起步时，additive n=4 的利用基本消失（per-agent support regret 从 0.414 降到 0.003，显著）。说明一大块利用来自"逐智能体重组 logged 动作"的起点。但 n=8 时，即使只用 logged 起点，ensemble 惩罚的规划结果仍然重复。
- 数据少（2k）时情况变差：joint n=8 的 ensemble-penalty regret 上升 0.57（显著）。

**Pilot 1 结论：H1–H4 都得到清楚支持。**

---

## 2. Pilot 2：多步 spread 任务（完整的离线 MBRL 流程）

N 个智能体、N 个地标，25 步，带碰撞惩罚。流程是：集成世界模型 → 通过模型反传训练去中心化策略 → 在真实环境评估。

两种数据：
- `assignment`：行为策略按排列分配目标，数据中没有拥挤。
- `independent`：每个智能体独立选目标。边缘分布相同，但数据里有拥挤。这是因果对照组。

惩罚：none、ens（MOPO 式）、agent_knn（逐智能体支撑）、joint_knn（联合支撑），以及后来加的 pair_knn（只看智能体对）。

规模：n=2,4,8 × 5 seeds × 2 数据 × 3 架构 × 5 惩罚，共 90 cells；pair_knn 另外 90 cells。

参照：oracle 用同一个策略类和优化器，在真实动力学上训练，回报约为 −2.8（n=2）、−2.8（n=4）、−2.9（n=8）。regret = oracle 回报 − 策略真实回报。注意：oracle 学到的是固定的 id→地标映射，没学到最优分配（见第 5 节局限）。

文件：`spread_grid/`、`spread_main_summary.txt`、`spread_main_analysis.txt`、`spread_pair/`。

### 2.1 世界模型诊断（拥挤状态，assignment 数据）
| arch | optimism n=2 / 4 / 8 | std/err n=2 / 4 / 8 |
|---|---|---|
| additive | 0.60 / 0.27 / 0.21 | 0.02 / 0.05 / 0.09 |
| joint | 0.09 / 0.12 / 0.19 | 0.64 / 0.30 / 0.31 |
| attention | 0.22 / 0.14 / 0.17 | 0.21 / 0.29 / 0.24 |

所有模型在没见过的拥挤状态上都偏乐观，而且 std 明显小于误差。→ **H1 在多步任务里也成立。**

换成 independent 数据后，attention 的拥挤状态误差接近 0，std/err 约为 1，校准变好。joint 在 n≤4 时也接近 0。

### 2.2 Regret vs oracle（assignment 数据）
| arch | penalty | n=2 | n=4 | n=8 |
|---|---|---|---|---|
| additive | none | 17.7 | 49.7 | 57.8 |
| additive | ens:1 | 16.9 | 29.6 | 34.4 |
| additive | joint_knn | 1.5 | 6.0 | 12.9 |
| additive | pair_knn | 1.1 | 34.3 | 35.9 |
| joint | none | 0.3 | 27.2 | 28.0 |
| joint | ens:1 | 0.9 | 2.5 | 7.3 |
| joint | joint_knn | 1.1 | 2.1 | 6.7 |
| joint | pair_knn | 1.1 | 2.0 | 8.4 |
| attention | none | 0.0 | −0.1 | 13.4 |
| attention | ens:1 | 0.8 | 0.6 | 4.8 |
| attention | joint_knn | 0.8 | 0.9 | 7.0 |
| attention | pair_knn | 0.8 | 0.6 | 12.1 |

### 2.3 Exploit gap 分解（117 cells，世界模型 117/117 复现成功）
exploit gap = 沿真实轨迹的单步模型误差 + 累积误差（compounding）。
- **累积误差基本为 0**（绝大多数在 ±0.5 内）。几乎全部 gap 来自单步误差。
- 单步误差几乎全部发生在**下一步真实状态是拥挤状态**的那些步上，n=8 时这些步接近 100%。
- 所以策略利用的是模型对"拥挤联合状态"的单步乐观，不是长 horizon 的误差累积。

### 2.4 Horizon 消融（H=5/10 vs 25，n=4,8，assignment）
- 短 horizon **不减少** regret，反而常常更差。例如 attention ens:1 在 n=4 时 regret 从 1.0 升到 8.0（H=5）。
- 原因之一是真实动力学下的短视成本本身就很大（H=5 的 oracle 比 H=25 差 10–19）。horizon-matched oracle 自己也大量拥挤，所以这个对照被短视混淆了，不能干净地分出模型误差那一部分。

---

## 3. 结果与研究方向是否匹配

假设编号按阶段 3 文档：H1 模型乐观且集成看不出，H2 偏差来自缺少联合覆盖，H3 利用随 n 增加，H4 逐智能体惩罚挡不住、联合支撑能挡但保守，H5 真实数据集。

**匹配得好的部分：**
1. H1（模型在 joint-OOD 上乐观、集成低估误差）：两个 pilot 都清楚支持，数字一致。
2. H3（n 越大越严重）：所有架构在 n=8 时都出问题，n≤4 时 joint/attention 基本正常。
3. 机制：分解结果表明，利用来自对拥挤联合状态的单步乐观，和"joint-action OOD"这个故事完全一致。
4. Pilot 1 中联合支撑约束完全解决问题。

**不完全匹配、需要你判断的部分：**
1. **H4（联合支撑能挡）在 spread 里只对 additive 干净成立。** joint_knn 并没有明显优于 ens。joint n=8 时 ens:1 是 7.3，joint_knn 是 6.7，差不多；attention n=8 时 joint_knn（7.0）反而比 ens（4.8）差。在多步任务里，联合支撑约束没能像 pilot 1 那样解决问题，n=8 时所有惩罚都还剩 5–10 的 regret。
2. **因果对照（independent 数据）结果是混合的：**
   - attention：n=8 时有拥挤数据，regret 下降（joint_knn 差 +3.1，p<0.001），符合假设。
   - joint：反而变差（ens:5 在 n=8 时差 −7.4，p=0.001）。数据里有拥挤并没有帮到 joint 模型，可能是同样数据量下有效覆盖变稀了。
   - additive：基本由模型容量决定（结构上无法表示交互），和 OOD 关系不大。additive 应该当作"容量不足"的对照，而不是 OOD 的证据。
3. **additive 偏差对 K、数据量不敏感这一点只在 pilot 1 中验证过。** spread 里没做数据量、K 等消融；只做了 horizon，而 horizon 被短视混淆了。
4. **pair_knn 不如 joint_knn。** additive 上 n=4/8 差很多（34.3 / 35.9 vs 6.0 / 12.9），joint 上差不多，attention n=8 更差。可能是对所有智能体对取平均稀释了信号，需要 λ 扫描后再下结论。

**建议的研究方向调整（供你决定）：**
- 主线论点建议定为"集成不确定性在 joint-OOD 上**校准失败**（排序正确但幅度严重低估），利用来自单步拥挤乐观"。这一点证据最强。
- "联合支撑约束能解决问题"在多步设定里证据不足，要么弱化这个说法，要么在 spread 里改进支撑约束再验证（例如支撑约束作用在拥挤相关的特征上，或者用更强的密度模型）。
- 如果要对外发表，需要补：spread 上的 K/数据量消融，以及一个更强的 oracle（能学会最优分配的策略类）。

---

## 4. 计算记录
- GPU：RTX 4080，PyTorch 2.6.0+cu124，venv `C:\Users\gengl\venvs\torch-cu124`。
- 主网格 5 seeds：约 10.6 h + 7 h；pair_knn：8.9 h；horizon 两个阶段约 1 h；gap 分解约 4 h。
- 所有运行 0 FAIL。中途修过的 bug：init_logged 的 reshape、stdout 管道断开导致丢结果、ablation chain 参数顺序、`-WaitPid 0` 死等。这些都在产生错误数据之前修好了，或者只丢了结果、没有写入错误数据。

## 5. 已知局限
1. Pilot 1 的 PyTorch 移植版已和 JAX 参考实现对比过（port_check/，n=2/4，2 seeds，结果一致），但只覆盖了 additive 和 joint。
2. spread 的 oracle 只学到固定映射，regret 是相对"同策略类能达到的最好"，不是真实最优。
3. duplicate_rate 在 n=8 时饱和；uncovered_frac 是中途加的，早期 cells 没有。
4. spread n=8 的部分 cell 只有 3–4 个 seed 有 uncovered_frac 等后加指标。
