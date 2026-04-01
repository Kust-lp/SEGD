from .component import *
from ._2DSE import SE
from .DENet import DENet
from .DRMs import DRM



class IRRestorationBackbone(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        num_res: int = 8,
        num_res_Dec: int = 4,
        num_res_evi = 6,
        in_dim: int = 128,
        num_classes: int = 3,
        hidden: int = 256,
        d_e = 128
    ):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, base_ch=base_ch, num_res=num_res)
        self.evi = DENet(in_ch = in_ch, base_ch=base_ch,
                                 num_res=num_res_evi, in_dim=in_dim,
                                 num_classes=num_classes, hidden=hidden
                                 )

        self.se_choose = SE()
        self.film = FiLM(self.encoder.out_ch, d_e=d_e, s_gamma=0.5, s_beta=0.1)
        self.CDRM =  DRM(in_c=self.encoder.out_ch, out_c=base_ch*2)
        self.BDRM =  DRM(in_c=self.encoder.out_ch, out_c=base_ch*2)
        self.NDRM =  DRM(in_c=self.encoder.out_ch, out_c=base_ch*2)
        self.decoder = Decoder(in_ch=base_ch*2, num_res=num_res_Dec)
        self.avgResdegarde = True

    def sigle_degarde(self, feat, deg):
        if torch.equal(deg, torch.tensor([1., 0., 0.], device=deg.device)):
            return self.CDRM(feat)
        if torch.equal(deg, torch.tensor([0., 1., 0.], device=deg.device)):
            return self.BDRM(feat)
        if torch.equal(deg, torch.tensor([0., 0., 1.], device=deg.device)):
            return self.NDRM(feat)

    def double_degarde(self, feat,  deg):
        if torch.equal(deg, torch.tensor([1., 1., 0.], device=deg.device)):
            f1 = self.BDRM(self.CDRM(feat))
            f2 = self.CDRM(self.BDRM(feat))
        if torch.equal(deg, torch.tensor([1., 0., 1.], device=deg.device)):
            f1 = self.NDRM(self.CDRM(feat))
            f2 = self.CDRM(self.NDRM(feat))
        if torch.equal(deg, torch.tensor([0., 1., 1.], device=deg.device)):
            f1 = self.NDRM(self.BDRM(feat))
            f2 = self.BDRM(self.NDRM(feat))
        f = (f1 + f2) / 2.0
        return f


    def three_degarde(self, feat):
        f_c = self.CDRM(feat)
        f_cbn = self.NDRM(self.BDRM(f_c))
        f_cnb = self.BDRM(self.NDRM(f_c))

        f_b = self.BDRM(feat)
        f_bcn = self.NDRM(self.CDRM(f_b))
        f_bnc = self.CDRM(self.NDRM(f_b))

        f_n = self.NDRM(feat)
        f_ncb = self.BDRM(self.CDRM(f_n))
        f_nbc = self.CDRM(self.BDRM(f_n))

        if self.avgResdegarde:
           return (f_cbn + f_cnb + f_bcn + f_bnc + f_ncb + f_nbc) / 6.0
        else:
            feats = torch.cat([f_cbn, f_cnb, f_bcn, f_bnc, f_ncb, f_nbc], dim=0)

            with torch.no_grad():
                choosed_id, weights = self.se_choose(feats.detach())
                weights = weights.detach()
            choose_feats = feats[choosed_id]
            return (choose_feats * weights).sum(dim=0).unsqueeze(0)



    def forward(self, x: torch.Tensor, label: torch.Tensor = None, zeta: float = 0.45) :
        with torch.no_grad():
            deg, _, _, p, S = self.evi(x)
            t = torch.tensor([zeta, zeta, zeta], device=deg.device)
            deg = torch.sigmoid(deg)
            deg_exist = (deg > t).float()
        # deg_exist = label[:,:3] # for train
        p = p.detach()
        S = S.detach()
        deg = deg.detach()
        deg_exist.detach()
        feat = self.encoder(x)
        feat = self.film(feat, p, S)

        feats = []
        for id, raw in enumerate(deg_exist):
            fi = feat[id].unsqueeze(0)

            if raw.sum().item() == 0.:
                raw = (deg[id] == deg[id].max()).float()
                fi = self.sigle_degarde(fi, raw)
            elif raw.sum().item()  == 1.:
                fi = self.sigle_degarde(fi, raw)
            elif raw.sum().item()  == 2.:
                fi = self.double_degarde(fi, raw)
            else:
                fi = self.three_degarde(fi)
            feats.append(fi)

        feats = torch.cat(feats, dim=0)
        output = self.decoder(feats)
        return output






