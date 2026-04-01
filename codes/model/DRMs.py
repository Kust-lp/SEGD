from .component import *
from .DENet import DENet
from .Restormer import Restormer




class DRM(nn.Module):
    def __init__(self,in_c = 64, out_c = 16):
        super().__init__()
        self.out_c = out_c
        self.mlp =  nn.Sequential(
            ConvGAct(in_c, out_c),
            ConvGAct(out_c, out_c)
        )
        self.Restormer = Restormer(in_c=out_c, out_c=out_c)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[1]!= self.out_c:
            x = self.mlp(x)
        z = self.Restormer(x)
        return x - z

class DRMsBackbone(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        num_res: int = 8,
        num_res_evi: int = 6,
        in_dim: int = 128,
        num_classes: int = 3,
        hidden: int = 256,
        d_e = 128,

    ):
        super().__init__()
        self.evi = DENet(in_ch=in_ch, base_ch=base_ch,
                                 num_res=num_res_evi, in_dim=in_dim,
                                 num_classes=num_classes, hidden=hidden,
                                 )

        self.encoder = Encoder(in_ch=in_ch, base_ch=base_ch, num_res=num_res)
        self.film = FiLM(self.encoder.out_ch, d_e=d_e, s_gamma=0.5, s_beta=0.1)
        self.expert = DRM(in_c=self.encoder.out_ch, out_c=base_ch*2)
        self.decoder = Decoder(in_ch=base_ch*2, num_res=num_res//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, _, p, S = self.evi(x)
        z = self.encoder(x)
        z = self.film(z, p, S)
        z = self.expert(z)
        ir_hat = self.decoder(z)
        return ir_hat
