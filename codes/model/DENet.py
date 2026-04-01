
from .component import *



class MultiHeadEvidentialBeta(nn.Module):

    def __init__(
        self,
        in_ch: int = 64,
        in_dim: int = 128,
        num_classes: int = 3,
        hidden: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.Convblock = nn.Sequential(
            ConvGAct(in_ch, in_ch//2, 3, 1, g=8),
            ConvGAct(in_ch//2, in_ch//2, 3, 1, g=8)
        )
        self.pool = GeMPool2d()

        self.class_heads = nn.Sequential(
            nn.Linear(in_ch // 2, in_dim),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),  # "contrast", "blur", "noise"
        )

        self.edl_head = nn.Sequential(
            nn.Linear(in_ch // 2, in_dim),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),  # e_pos, e_neg
        )


    def forward(self, x: torch.Tensor):

        x = self.Convblock(x)
        v = self.pool(x)  # (B,64)

        deg = self.class_heads(v)  # (B,3)
        strength = self.edl_head(v)  # (B,2)

        alpha = F.softplus(strength[:, 0]) + 1.0
        beta = F.softplus(strength[:, 1]) + 1.0
        S = alpha + beta
        p_hat = alpha / S
        return deg, alpha, beta, p_hat, S


class DENet(nn.Module):
    def __init__(self,
        in_ch: int = 1,
        base_ch: int = 16,
        num_res: int = 6,
        in_dim: int = 128,
        num_classes: int = 3,
        hidden: int = 256,
        ):
        super(DENet, self).__init__()
        self.encoder = Encoder(in_ch=in_ch, base_ch=base_ch, num_res=num_res)
        self.evi = MultiHeadEvidentialBeta(self.encoder.out_ch, in_dim, num_classes, hidden)

    def forward(self, x: torch.Tensor) :
        x = self.encoder(x)
        deg, alpha, beta, p_hat, S = self.evi(x)
        return deg, alpha, beta, p_hat, S
