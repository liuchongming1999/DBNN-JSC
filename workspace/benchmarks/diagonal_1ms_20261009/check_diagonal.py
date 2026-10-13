"""Check the added diagonal using the same convolution and physical state."""
import json
import torch
from train_dbnn_diagonal import VariableDrive, random_state, fft_length

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device('cuda:0')
state = random_state(291049)
n = 64
x = torch.zeros((2, 500, n), device=device)
x[:, :3, ::7] = 1
xf = torch.fft.rfft(x, n=fft_length(n))
zero = VariableDrive(state, device, 1., False)
full = VariableDrive(state, device, 1., True)
with torch.no_grad():
    initial_gap = float((zero(xf, n) - full(xf, n)).abs().max())
assert initial_gap == 0
full(xf, n).sum().backward()
zero(xf, n).sum().backward()
full_gradient = full.weight.grad.diag()
assert full_gradient[:3].abs().min() > 0
assert zero.weight.grad.diag().abs().max() == 0
with torch.no_grad():
    full.weight.diagonal()[:3].fill_(0.02)
    t = torch.arange(n, device=device, dtype=torch.float32)
    tr, td = full.log_tau_rise.exp()[:, None], full.log_tau_decay.exp()[:, None]
    k = full.omega[:, None] * (1 - torch.exp(-t / tr)) * torch.exp(-t / td)
    y = torch.fft.irfft(xf * torch.fft.rfft(k, n=fft_length(n))[None],
                        n=fft_length(n))[..., :n].transpose(1, 2)
    difference = full(xf, n) - zero(xf, n)
    expected = (y.square() * full.weight.diag()).sum(-1)
    difference_gap = float((difference - expected).abs().max())
    assert difference_gap < 5e-5, difference_gap
    physical = full.physical_state({'v_th': -55., 'reset_amp': 15., 'tau_reset': 10.})
    saved = physical['bilinear.weight'][0].diag()
    assert saved[:3].abs().min() > 0
print(json.dumps({'passed': True, 'initial_zero_full_gap_mv': initial_gap,
                  'quadratic_diagonal_identity_max_gap_mv': difference_gap,
                  'diagonal_gradient_nonzero': True,
                  'serialized_diagonal_retained': True}))
