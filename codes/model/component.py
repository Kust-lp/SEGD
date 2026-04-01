import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvGAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, g=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2 )
        self.bn = nn.GroupNorm(min(g, out_ch), num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        assert in_ch == out_ch, "in_ch must equal to out_ch"
        self.out_ch = out_ch
        self.res = nn.Sequential(
            ConvGAct(in_ch, out_ch, k, s),
            ConvGAct(out_ch, out_ch, k, s),
        )
    def forward(self, x):
        return x + self.res(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.f1 = nn.Conv2d(ch, ch // r, kernel_size=1)
        self.f2 = nn.Conv2d(ch // r, ch, kernel_size=1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.f1(w))
        w = torch.sigmoid(self.f2(w))
        return x * w


class GeMPool2d(nn.Module):

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        self.eps = eps
        if learn_p:
            self.raw_p = nn.Parameter(torch.tensor(float(p_init)))
        else:
            self.register_buffer("raw_p", torch.tensor(float(p_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,C,H,W)
        # p > 0 with a tiny floor for stability
        p = F.softplus(self.raw_p) + 1e-6                      # scalar tensor on same device/dtype as raw_p
        x = x.clamp_min(self.eps).pow(p)                       # (B,C,H,W)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)           # (B,C,1,1)
        return x.view(x.size(0), x.size(1))                    # (B,C)

class EvidenceTower(nn.Module):
    def __init__(self, d_e=32, eps=1e-5, mom=0.05):
        super().__init__()
        self.eps = eps
        self.mom = mom
        self.mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(inplace=True),
            nn.Linear(32, d_e), nn.ReLU(inplace=True),
        )
        self.register_buffer('mu', torch.zeros(2))  # running mean
        self.register_buffer('sig', torch.ones(2))  # running std
        self.Wp = nn.Parameter(torch.ones(d_e))
        self.Ws = nn.Parameter(torch.ones(d_e))
        self.bias = nn.Parameter(torch.zeros(d_e))

    @torch.no_grad()
    def _update_ema(self, z):
        batch_mu = z.mean(dim=0)
        batch_var = z.var(dim=0, unbiased=False).clamp_min(self.eps)
        batch_sig = batch_var.sqrt()
        self.mu = (1 - self.mom) * self.mu + self.mom * batch_mu
        self.sig = (1 - self.mom) * self.sig + self.mom * batch_sig

    def _standardize(self, z):
        return (z - self.mu) / (self.sig + self.eps)

    def forward(self, p, S):
        z_p = torch.logit(p, eps=1e-8)
        z_S = torch.log(S+1e-8)
        z = torch.stack([z_p, z_S], dim=1)
        if self.training:
            self._update_ema(z.detach())
        z = self._standardize(z)
        h = self.mlp(z)                                   # (B,d_e)
        mono = F.softplus(self.Wp)[None,:]*z[:,0:1] + F.softplus(self.Ws)[None,:]*z[:,1:2] + self.bias
        return h + mono                                   # (B,d_e)



class FiLM(nn.Module):
    def __init__(self, ch, d_e=128, s_gamma=0.5, s_beta=0.1):
        super().__init__()
        self.s_gamma, self.s_beta = s_gamma, s_beta
        self.et = EvidenceTower(d_e=d_e)
        self.head = nn.Linear(128, ch*2)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, feat, p, S):
        B, C = feat.size(0), feat.size(1)
        e = self.et(p, S)                                  # (B,d_e)
        raw = self.head(e).view(B, 2, C)  # (B,T,2,C)
        gam = 1 + self.s_gamma * torch.tanh(raw[:, 0, :])
        bet = self.s_beta  * torch.tanh(raw[:,  1, :])
        gam = gam.view(B, C, 1, 1)
        bet = bet.view(B, C, 1, 1)
        return feat * gam + bet

class Decoder(nn.Module):
    def __init__(
        self,
            in_ch: int = 16,
            num_res: int = 2,
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*2, 1, 1),
            *[ResBlock(in_ch*2,in_ch*2) for _ in range(num_res)],
            nn.Conv2d(in_ch * 2, in_ch, 3, 1, 1),
            nn.Conv2d(in_ch, 1, 1,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        return z

class Encoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        num_res: int = 4,
    ):
        super().__init__()
        self.out_ch = base_ch*4
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, 1, 1),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
        )
        self.body = nn.Sequential(
            ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, base_ch*2, 1, 1),
            ResBlock(base_ch*2, base_ch*2),
            nn.Conv2d(base_ch*2, base_ch*4, 1, 1),
            *[ResBlock(base_ch*4, base_ch*4) for _ in range(num_res)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.body(z)
        return z