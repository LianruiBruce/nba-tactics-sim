# 真实数据工具：命令和示例结果

下面每一步都给出实际命令和真实运行的输出（模型训练截至 2026-10-25，2025-26 赛季数据）。时间是 4 核云主机上的耗时。

## 0. 准备数据

```bash
pip install -e ".[dev,data]" scikit-learn scipy
git fetch origin nba-data && git worktree add ../nba-data origin/nba-data
ln -s ../../nba-data/data/nba data/nba        # Windows 上可以直接把 ../nba-data/data/nba 复制到 data/nba
python scripts/build_dataset.py               # 约 3 分钟 -> data/processed/<赛季>/possessions.parquet
```

想重新抓最新数据时，在自己电脑上运行 `python scripts/fetch_nba.py --seasons 2026-27`（stats.nba.com 常屏蔽云服务器）。

`data/models/realmodel.pkl` 已经提交在仓库里，只用推演和球探工具的话，可以跳过第 1、2 步。

## 1. 回测（可选，约 30 分钟）

```bash
python scripts/backtest.py                        # 用 2026-01-01 之前的比赛训练，预测 2025-26 剩下的 734 场
python scripts/backtest.py --cutoff 2025-10-01    # 赛季前：预测 2025-26 全部 1230 场
```

输出（节选，`--cutoff 2026-01-01`）：

```
            game_margin_rmse  game_margin_corr  winner_accuracy  lineups  lineup_net_corr  lineup_net_rmse
context              16.9015            0.2251           0.5654    119.0           0.3638          14.8231
team                 16.5505            0.2635           0.6035    119.0           0.1672          14.8825
rapm                 15.0839            0.4822           0.7112    119.0           0.3982          13.3292
real                 15.0360            0.4850           0.7071    119.0           0.4701          12.9571
real+resid           15.0008            0.4867           0.7112    119.0           0.4530          12.9880
game margin RMSE, real+resid minus rapm: -0.083 (95% CI -0.177 to +0.012)
```

`real` 是我们的球员模型，`rapm` 是业界常用的基线。整场分差两者打平，五人组净效率我们更准（误差 12.96 对 13.33，相关 0.47 对 0.40）。

## 2. 训练正式模型（十几分钟）

```bash
python scripts/fit_model.py      # 用全部数据训练 -> data/models/realmodel.pkl
```

## 3. 阵容推演 `whatif.py`（几秒）

```bash
python scripts/whatif.py \
    --a "Gilgeous-Alexander, Jalen Williams, Dort, Holmgren, Hartenstein" \
    --b "Doncic, Reaves, LeBron James, Hachimura, Ayton" \
    --home a --try "Caruso, Cason Wallace"
```

球员名可以只写姓或名字的一部分，有重名时会报错并列出候选人。`--home a` 表示 A 是主场；`--try` 是要试着换进 A 阵容的候选人。

输出：

```
A: Shai Gilgeous-Alexander, Jalen Williams, Luguentz Dort, Chet Holmgren, Isaiah Hartenstein
B: Luka Dončić, Austin Reaves, LeBron James, Rui Hachimura, Deandre Ayton

             A 进攻   B 进攻
每回合得分       1.211  1.082
失误/机会       0.104  0.140
罚球回合/机会     0.066  0.092
篮下 出手占比     0.280  0.239
短中距 出手占比    0.276  0.245
长两分 出手占比    0.091  0.110
底角三分 出手占比   0.087  0.133
弧顶三分 出手占比   0.266  0.273
篮下 命中率      0.725  0.664
短中距 命中率     0.478  0.457
长两分 命中率     0.456  0.427
底角三分 命中率    0.411  0.398
弧顶三分 命中率    0.375  0.355
前场篮板率（投丢后）  0.275  0.228

A 对 B 的净效率：+12.9 分/100 回合

A 的出手分配：Shai Gilgeous-Alexander 29%，Jalen Williams 23%，Chet Holmgren 20%，Luguentz Dort 17%，Isaiah Hartenstein 12%
B 的出手分配：Luka Dončić 27%，LeBron James 22%，Austin Reaves 18%，Rui Hachimura 18%，Deandre Ayton 15%

换人推演（把 A 的某人换成候选人后的净效率变化）：
           换上                      换下  净效率变化
  Alex Caruso          Jalen Williams    7.2
  Alex Caruso           Luguentz Dort    6.1
  Alex Caruso      Isaiah Hartenstein    2.4
Cason Wallace          Jalen Williams    1.5
  ...
Cason Wallace Shai Gilgeous-Alexander   -8.4
```

怎么读：雷霆这五人对湖人这五人，模型预计每 100 回合净胜 12.9 分。主要来自失误更少（每次机会 10.4% 对 14.0%）和篮下命中更高。换人表说的是"只针对这个对手"：用卡鲁索换下杰伦·威廉姆斯，预计再多 7.2 分。换人推演只算攻防效率，不考虑体能和犯规。

## 4. 对手球探报告 `scout.py`（约 1 分钟）

```bash
python scripts/scout.py --team OKC --season 2025-26          # -> data/reports/OKC_2025-26.md
python scripts/scout.py --team BOS --season 2026-27 --since 2026-10-01
```

完整示例：[data/reports/OKC_2025-26.md](../data/reports/OKC_2025-26.md)。报告包含球队效率和排名、投篮结构对比联盟、NBA.com 战术类型效率、常用五人组（实际和模型预期净效率）、关键球员评分，以及自动生成的比赛策略。节选：

| 指标 | 数值 | 联盟排名 |
| --- | --- | --- |
| 进攻效率（每 100 回合） | 117.1 | 7 |
| 防守效率 | 106.3 | 1 |
| 净效率 | +10.8 | 1 |

> - 他们在底角三分的防守最弱：允许命中率 40.0%，比联盟高 +1.4 个百分点，多找这个区域的出手。
> - 他们最依赖挡拆持球（占 17.1%，每回合 0.977 分），重点限制。
> - 他们最弱的常用阵容是 S. Gilgeous-Alexander, L. Dort, I. Joe, J. Williams, C. Wallace（净效率 -20.9），这套人在场时加快节奏。

## 5. 用真实球员打一场模拟比赛 `simulate.py --real`（约 25 秒）

```bash
python scripts/simulate.py --real --quarters 4 \
    --a "Gilgeous-Alexander, Jalen Williams, Dort, Holmgren, Hartenstein" \
    --b "Brunson, Mikal Bridges, Josh Hart, Anunoby, Towns" --out replays/okc_nyk.json
python -m http.server     # 打开 http://localhost:8000/viewer/ ，选择 replays/okc_nyk.json 看回放
```

输出（一场，seed 42）：

```
Gilgeous-Alexander / Williams / Dort / Holmgren / Hartenstein  137
  player                 PTS  FG     3P    FT    REB AST STL BLK TOV PF
  Shai Gilgeous-Alexander  47  19-32   3-6   6-7   19  14   4   0   4  5
  Jalen Williams          20   9-20   1-5   1-1    8   9   3   1   4  5
  Luguentz Dort           27   8-15   8-15  3-3    9   3   1   0   0  0
  Chet Holmgren           22  10-18   1-4   1-2    7   3   0   1   5  4
  Isaiah Hartenstein      21   9-13   0-0   3-5   10   0   3   1   0  3
Brunson / Bridges / Hart / Anunoby / Towns  131
  player                 PTS  FG     3P    FT    REB AST STL BLK TOV PF
  Jalen Brunson           45  18-35   9-20  0-0   20   9   1   0   3  5
  Mikal Bridges           19   6-18   3-10  4-7    6   7   5   0   4  5
  Josh Hart               12   4-12   3-9   1-2    4   1   1   0   2  2
  OG Anunoby              28   9-21   7-14  3-3   14   3   4   2   3  3
  Karl-Anthony Towns      27  10-17   2-8   5-5    8   1   0   4   3  3
possessions [109, 109], frames 31813
```

注意：引擎里没有替补和体能，五个人打满 48 分钟，所以个人数据会比真实比赛高很多（后卫的篮板也偏多）。单场随机性很大，比较两套阵容请看第 6 步的多场统计或第 3 步的模型推演。

## 6. 检验引擎和真实数据是否一致 `engine_check.py`（约 25 分钟）

```bash
python scripts/engine_check.py --pairs 15 --games 10   # 各队最常用五人组两两互打 150 场
```

输出（节选）：

```
corr(engine, model) = 0.54; corr(engine, actual unit gap) = 0.48; corr(model, actual unit gap) = 0.43

per matchup side, engine vs model (slope of engine on model; 1 = engine keeps the model's spread):
  ppp             slope  0.80  corr  0.63
  tov per chance  slope  0.86  corr  0.71
  fg pts per fga  slope  0.93  corr  0.70
  rim fg          slope  1.04  corr  0.68
  three fg        slope  1.15  corr  0.68
```

斜率接近 1 表示引擎保留了模型预测的强弱差距。
