"""Check critical label/mask gradients and the original reset recurrence."""
import sys
from pathlib import Path
import numpy as np
import torch
from train_events import event_loss, sequential_reset

torch.set_num_threads(1)
voltage = torch.tensor([[-60., -54., -100., 20., -60., -60.]], requires_grad=True)
# Positive reference event at sample 0; a true false-trigger candidate at 1;
# sample 3 is a padded point and must contribute no gradient.
positive = torch.tensor([0])
negative_valid = torch.tensor([[False, True, True, False, True, True]])
p, n = event_loss(voltage, positive, negative_valid, -55.)
(p+n).backward()
assert voltage.grad[0, 0] < 0 and voltage.grad[0, 1] > 0
assert voltage.grad[0, 3] == 0 and voltage.grad[0, 2] == 0
assert torch.isfinite(voltage.grad).all()
sys.path.insert(0, '/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/channel_ablation_20260929')
from train_channel_ablation import reset
rng = np.random.default_rng(410049)
worst = 0.
for length in (1001, 4001, 16001):
    drive = -60.+rng.normal(0., 4., length)
    for dt in (1., .5, .25):
        a, ea = reset(drive, -56., 8., 16./dt)
        b, eb = sequential_reset(drive, -56., 8., 16./dt)
        assert np.array_equal(ea, eb)
        worst = max(worst, float(np.max(abs(a-b))))
assert worst < 1e-8
print('PASS: event gradients, padding exclusion, reset events and voltage; max reset error', worst)
