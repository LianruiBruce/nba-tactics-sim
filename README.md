# nba-tactics-sim

多智能体 NBA 战术模拟器，同时也是强化学习环境。每个球员都是一个 agent，数值采用 NBA 2K 的属性体系。比赛引擎负责规则和裁决（2/3 分、24 秒、投篮命中、传球抢断、篮板）。长期目标是让 agent 通过强化学习学出新的战术。

设计方案：https://claude.ai/artifact/Cyp5Ju3peKZyprNP4VwUm5

## 快速开始

```bash
pip install -e ".[dev]"
python scripts/simulate.py                # 一攻一防两个回合，输出 replays/sample.json
python scripts/simulate.py --batch 1000   # 批量统计每回合得分、结果分布、出手区域
pytest
```

看回放：在仓库根目录运行 `python -m http.server`，然后打开 http://localhost:8000/viewer/ 。也可以在页面里选择本地的回放 JSON 文件。

## 结构

| 路径 | 内容 |
| --- | --- |
| `nbasim/court.py` | 半场坐标（英尺）、三分线、出手区域、15 个战术点位 |
| `nbasim/players.py` | 2K 属性分组的 Pydantic 模型，加载球员与阵容 |
| `nbasim/engine.py` | 10 Hz 确定性引擎：移动与碰撞、计时、传球/抢断、投篮、篮板 |
| `nbasim/macro.py` | 宏动作空间（跑位、传球、投篮、突破、掩护、盯人）、动作掩码、底层控制器 |
| `nbasim/env.py` | PettingZoo `ParallelEnv`，一个 episode 就是一个回合 |
| `nbasim/agents/scripted.py` | 手写基线策略：挡拆 + 人盯人（阶段 1 换成剧本解释器） |
| `nbasim/replay.py` | 回放日志格式 |
| `viewer/index.html` | Canvas 回放器 |
| `data/players/nba2k26.yaml` | 10 名真实球员的 2K 数值 |
| `data/lineups.yaml` | 测试用阵容 |

## RL 环境

```python
from nbasim.env import HalfCourtEnv
from nbasim.players import load_lineups

L = load_lineups()
env = HalfCourtEnv(L["mixed"], L["west"])
obs, infos = env.reset(seed=0)
# obs[agent] = {"observation": float32[147], "action_mask": int8[31]}
```

- 智能体：`off_0..4` 进攻，`def_0..4` 防守，顺序为 PG、SG、SF、PF、C。
- 一步 = 0.5 秒（引擎跑 5 个 tick）。
- 奖励：回合结束时进攻方 +得分，防守方 −得分，队内共享。

## 现状与已知问题（阶段 0）

- `nba2k26.yaml` 里的数值是凭记忆填的近似值，还没有核对。校准引擎之前要对照 2kratings.com 更正。
- 命中率、抢断率、篮板模型只是占位公式，还没有用联盟数据校准。脚本基线目前出手过于集中在篮下。
- 规则还没有做犯规、罚球、三秒、出界。
- 性能：单核约 860 env 步/秒（约 4300 tick/秒），目标是 5000 步/秒，碰撞和观测构建还需要向量化。
