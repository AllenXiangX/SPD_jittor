import jittor as jt
import jittor.nn as nn
from jittor.contrib import concat
from .utils import MLP_Res, build_latent_flow, \
    reparameterize_gaussian, gaussian_entropy, \
    standard_normal_logprob, truncated_normal_, FurthestPointSampler
from loss_functions import chamfer_l2 as chamfer
from .SPD import SPD

class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose(dim_feat, 128, (num_pc, 1), bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def execute(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat.unsqueeze(3)).squeeze(3)  # (b, 128, 256)
        x1 = self.mlp_1(concat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(concat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        out_points = self.mlp_4(x3)  # (b, 3, 256)
        return out_points

class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_p0=512,
                 radius=1, bounding=True, up_factors=None):
        super(Decoder, self).__init__()
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_p0)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def execute(self, feat):
        """
        Args:
            feat: Tensor, (b, dim_feat)
            partial_coarse: Tensor, (b, n_coarse, 3)
        """
        feat = feat.unsqueeze(-1)
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1)  # (B, num_pc, 3)
        arr_pcd.append(pcd)
        feat_prev = None

        for upper in self.uppers:
            pcd, feat_prev = upper(pcd, feat, feat_prev)
            arr_pcd.append(pcd)

        return arr_pcd

class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)



    def execute(self, x):
        x = x.transpose(1, 2)
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = jt.max(x, 2, keepdims=True)
        x = x.view(-1, 512)

        m = nn.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = nn.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = nn.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = nn.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        return m, v


class ModelVAE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        dim_feat = kwargs.get('dim_feat', 128)
        num_p0 = kwargs.get('num_p0', 512)
        radius = kwargs.get('radius', 1)
        bounding = kwargs.get('bounding', True)
        up_factors = kwargs.get('up_factors', [2, 2])
        args = kwargs.get('args', None)
        self.encoder = PointNetEncoder(zdim=dim_feat)
        self.flow = build_latent_flow(args)
        self.decoder = Decoder(dim_feat=dim_feat, num_p0=num_p0,
                               radius=radius, up_factors=up_factors, bounding=bounding)
        self.fps_sampler_512 = FurthestPointSampler(512)

    def get_loss(self, x, kl_weight, writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        # print(x.size())
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)

        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)  # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, jt.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdims=True)  # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)

        # Negative ELBO of P(X|z)
        p1, p2, p3 = self.decoder(z)

        x_512 = self.fps_sampler_512(x)

        cd_1 = chamfer(p1, x_512)

        cd_3 = chamfer(p3, x)

        loss_recons = cd_1 + cd_3  # + emd_1 + emd_3  # + cd_2

        # Loss
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = loss_recons
        loss = kl_weight * (loss_entropy + loss_prior) + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', loss_entropy.item(), it)
            writer.add_scalar('train/loss_prior', loss_prior.item(), it)
            writer.add_scalar('train/loss_recons', loss_recons.item(), it)
            writer.add_scalar('train/z_mean', z_mu.mean().item(), it)
            writer.add_scalar('train/z_mag', z_mu.abs().max().item(), it)
            writer.add_scalar('train/z_var', (0.5 * z_sigma).exp().mean().item(), it)

        return loss

    def sample(self, w, truncate_std=None):
        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.decoder(z)[-1]
        return samples