# Joint-action exploitation pilot

Question: in model-based offline MARL, when each agent's action was seen in the
data but the combination was not, is the learned model confidently wrong, and do
ensemble-disagreement penalties miss it?

`pilot_assignment.py` (JAX) and `pilot_assignment_torch.py` (PyTorch, use this on
a Windows GPU) run the same one-step assignment game for 2 to 8 agents and three
reward-model architectures (joint MLP, per-agent attention, per-agent additive).

```
pip install torch scipy numpy
python research/joint_ood/pilot_assignment_torch.py --out research/joint_ood/results.json
python research/joint_ood/summarize.py research/joint_ood/results.json
```
