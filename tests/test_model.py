import torch

from kagrued.model import AttentionGRUEncoderDecoder


def test_shapes_and_attention_weights():
    model = AttentionGRUEncoderDecoder(n_features=5, horizon=4, hidden=16)
    x = torch.randn(8, 12, 5)
    y, attn = model(x, return_attention=True)
    assert y.shape == (8, 4)
    assert attn.shape == (8, 4, 12)
    torch.testing.assert_close(attn.sum(-1), torch.ones(8, 4))


def test_gradients_reach_encoder():
    model = AttentionGRUEncoderDecoder(n_features=3, horizon=2, hidden=8)
    model(torch.randn(4, 6, 3)).sum().backward()
    assert all(p.grad is not None for p in model.encoder.parameters())


def test_learns_a_sine_wave():
    torch.manual_seed(0)
    t = torch.arange(400, dtype=torch.float32)
    s = 0.5 + 0.4 * torch.sin(2 * torch.pi * t / 25)
    lookback, horizon = 20, 3
    X = torch.stack([s[i : i + lookback] for i in range(300)]).unsqueeze(-1)
    Y = torch.stack([s[i + lookback : i + lookback + horizon] for i in range(300)])
    model = AttentionGRUEncoderDecoder(1, horizon, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first
