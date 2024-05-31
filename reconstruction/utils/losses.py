import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
from utils.util import ssim
from collections import defaultdict, OrderedDict
import os
from torchvision.models import vgg19, VGG19_Weights

class AELoss(nn.Module):
    def __init__(self, beta=0, grad_pen_weight=0, grad_flag=False):
        super(AELoss, self).__init__()
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight
        self.grad_flag = grad_flag

    def grad_pen_loss(self, embedding, y_pred):

        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        embedding = net_out['z']
        
        # Reconstruction loss
        recon_loss = (net_in - x_hat) ** 2
        
        # Embedding loss
        embedding_loss = self.beta * (embedding ** 2).mean()
        
        # Total loss
        total_loss = recon_loss + embedding_loss
        
        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(embedding, x_hat)
        total_loss += grad_penalty

        if anomaly_score:
            if self.grad_flag:
                grad = torch.abs(torch.autograd.grad(total_loss.mean(), net_in)[0])
                if keepdim:
                    return torch.mean(grad, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty
                else:
                	return torch.mean(grad, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
            else:
            	if keepdim:
                    return torch.mean(total_loss, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty
            	else:
                	return torch.mean(total_loss, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
        else:
            return total_loss.mean(), recon_loss.mean(), embedding_loss, grad_penalty

class SSIMLoss(nn.Module):
    def __init__(self, win_size=11, beta=0, grad_pen_weight=0):
        super(SSIMLoss, self).__init__()
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight
        self.win_size = win_size

    def grad_pen_loss(self, embedding, y_pred):
        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)
        
    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']

        net_in = ((net_in + 1) / 2.0).clamp(0., 1.)
        x_hat = ((x_hat + 1) / 2.0).clamp(0., 1.)  # Normalize to [0, 1] for computing SSIM.
        embedding = net_out['z']
        embedding_loss = self.beta * (embedding ** 2).mean()
        
        loss = 1. - ssim(net_in, x_hat, data_range=1., size_average=False, win_size=self.win_size)
        total_loss = loss + embedding_loss
        
        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(embedding, x_hat)
        total_loss += grad_penalty
        if anomaly_score:
            if keepdim:
                return torch.mean(F.interpolate(loss, size=net_in.shape[-2:], mode='bilinear'), dim=[1], keepdim=True), loss, embedding_loss, grad_penalty
            else:
                return torch.mean(loss, dim=[1, 2, 3]), loss, embedding_loss, grad_penalty
        else:
            return loss.mean(), loss.mean(), embedding_loss, grad_penalty


class L1Loss(nn.Module):
    def __init__(self, beta=0, grad_pen_weight=0, grad_flag=False):
        super(L1Loss, self).__init__()
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight
        self.grad_flag = grad_flag

    def grad_pen_loss(self, embedding, y_pred):
        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        embedding = net_out['z']
        
        # Reconstruction loss
        recon_loss = torch.abs(net_in - x_hat)
        
        # Embedding loss
        embedding_loss = self.beta * (embedding ** 2).mean()
        
        # Total loss
        total_loss = recon_loss + embedding_loss
        
        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(embedding, x_hat)
        total_loss += grad_penalty

        if anomaly_score:
            if self.grad_flag:
                grad = torch.abs(torch.autograd.grad(total_loss.mean(), net_in)[0])
                if keepdim:
                    return torch.mean(grad, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty
                else:
                    return torch.mean(grad, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
            else:
            	if keepdim:
                    return torch.mean(total_loss, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty
            	else:
                    return torch.mean(total_loss, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
        else:
            return total_loss.mean(), recon_loss.mean(), embedding_loss, grad_penalty



# ----------- AE-U Loss ----------- #
class AEULoss(nn.Module):
    def __init__(self, beta= 0, grad_pen_weight = 0):
        super(AEULoss, self).__init__()
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight
        
    def grad_pen_loss(self, embedding, y_pred):
        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)
        
    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, log_var = net_out['x_hat'], net_out['log_var']
        recon_loss = torch.abs(net_in - x_hat)
        embedding = net_out['z']
        
        # Embedding loss
        embedding_loss = self.beta * (embedding ** 2).mean()
        
        loss1 = torch.exp(-log_var) * recon_loss

        loss = loss1 + log_var + embedding_loss

        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(embedding, x_hat)
        loss += grad_penalty
        
        if anomaly_score:
            return torch.mean(loss1, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty if keepdim else torch.mean(loss1, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
        else:
            return loss.mean(), recon_loss.mean().item(), log_var.mean().item()


# ----------- MemAE Loss ----------- #
class MemAELoss(nn.Module):
    def __init__(self, beta=0, grad_pen_weight=0):
        super(MemAELoss, self).__init__()
        self.entropy_loss_weight = 0.0002
        self.eps = 1e-12
        self.beta= beta
        self.grad_pen_weight = grad_pen_weight

    def grad_pen_loss(self, embedding, y_pred):
        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)
        
    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, att, z = net_out['x_hat'], net_out['att'], net_out['z']
        recon_loss = (net_in - x_hat) ** 2
        entro_loss = self.entropy_loss(att)

        embedding_loss = self.beta * (z ** 2).mean()
        
        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(z, x_hat)
        
        loss = recon_loss.mean() + self.entropy_loss_weight * entro_loss + grad_penalty + embedding_loss
        
        if anomaly_score:
            return torch.mean(recon_loss, dim=[1], keepdim=True), entro_loss, embedding_loss, grad_penalty if keepdim else torch.mean(recon_loss, dim=[1, 2, 3]), entro_loss, embedding_loss, grad_penalty
        else:
            return loss, recon_loss.mean().item(), entro_loss.item(), embedding_loss, grad_penalty

    def entropy_loss(self, x):
        x = self.feature_map_permute(x)
        b = x * torch.log(x + self.eps)
        b = -1. * b.sum(dim=1)
        return b.mean()

    def feature_map_permute(self, input):
        s = input.data.shape
        l = len(s)

        # permute feature channel to the last:
        # NxCxDxHxW --> NxDxHxW x C
        if l == 2:
            x = input  # NxC
        elif l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        x = x.contiguous()
        # NxDxHxW x C --> (NxDxHxW) x C
        x = x.view(-1, s[1])
        return x


# ----------- VAE Loss ----------- #
class VAELoss(nn.Module):
    def __init__(self, kl_weight=0.005, beta=0, grad_pen_weight=0, grad=None):
        super(VAELoss, self).__init__()
        # grad in ['elbo', 'rec', 'kl', 'combi']
        self.kl_weight = kl_weight
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight
        self.grad = grad
        
    def grad_pen_loss(self, z, net_out):
        x_hat = net_out['x_hat']
        grad = torch.autograd.grad(outputs=(x_hat ** 2).mean(), inputs=z, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat, mu, log_var = net_out['x_hat'], net_out['mu'], net_out['log_var']
        z = net_out['z'] 
        recon_loss = (net_in - x_hat) ** 2
        kl_loss = torch.mean(-0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=1)
        
        embedding_loss = self.beta * (z ** 2).mean()
        
        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(z, net_out)
        
        loss = recon_loss.mean() + self.kl_weight * kl_loss.mean() + embedding_loss + grad_penalty
        

        if anomaly_score:
            if self.grad == 'elbo':
                grad = torch.abs(torch.autograd.grad(loss, net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty if keepdim else torch.mean(grad, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
            elif self.grad == 'rec':
                grad = torch.abs(torch.autograd.grad(recon_loss.mean(), net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty if keepdim else torch.mean(grad, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
            elif self.grad == 'kl':
                grad = torch.abs(torch.autograd.grad(kl_loss.mean(), net_in)[0])
                return torch.mean(grad, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty if keepdim else torch.mean(grad, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
            elif self.grad == 'combi':
                kl_grad = torch.abs(torch.autograd.grad(kl_loss.mean(), net_in)[0])
                combi = recon_loss * kl_grad
                return torch.mean(combi, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty if keepdim else torch.mean(combi, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
            else:
                return torch.mean(recon_loss, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty if keepdim \
                    else torch.mean(recon_loss, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
        else:
            return loss, recon_loss.mean().item(), kl_loss.mean().item(), embedding_loss, grad_penalty


class GANomalyLoss(nn.Module):
    def __init__(self, beta=0, grad_pen_weight=0):
        super(GANomalyLoss, self).__init__()
        self.w_adv = 1
        self.w_rec = 50
        self.w_enc = 1
        self.grad_pen_weight = grad_pen_weight
        self.beta = beta
        
    def forward(self, net_in, net_out, mode='g', anomaly_score=False):
        z = net_out['z']
        embedding_loss = self.beta * (z ** 2).mean()
        

        if anomaly_score:
            z_hat = net_out['z_hat']
            return torch.mean((z - z_hat) ** 2, dim=1)
        else:
            if mode == 'g':
                x_hat, z, z_hat, feat_real, feat_fake = \
                    net_out['x_hat'], net_out['z'], net_out['z_hat'], net_out['feat_real'], net_out['feat_fake']
                loss_adv = torch.mean((feat_real - feat_fake) ** 2)
                loss_rec = torch.mean((net_in - x_hat) ** 2)
                loss_enc = torch.mean((z - z_hat) ** 2)

                loss_g = self.w_adv * loss_adv + self.w_rec * loss_rec + self.w_enc * loss_enc + embedding_loss
                return loss_g, loss_adv.item(), loss_rec.item(), loss_enc.item()
            else:
                l_bce = nn.BCELoss()
                pred_real, pred_fake_detach = net_out['pred_real'], net_out['pred_fake_detach']
                real_label = torch.ones(size=(pred_real.shape[0],), dtype=torch.float32).cuda()
                fake_label = torch.zeros(size=(pred_fake_detach.shape[0],), dtype=torch.float32).cuda()
                loss_d = (l_bce(pred_real, real_label) + l_bce(pred_fake_detach, fake_label)) * 0.5
                return loss_d


class AELoss(nn.Module):
    def __init__(self, beta=0, grad_pen_weight=0, grad_flag=False):
        super(AELoss, self).__init__()
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight
        self.grad_flag = grad_flag

    def grad_pen_loss(self, embedding, y_pred):

        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        embedding = net_out['z']
        
        # Reconstruction loss
        recon_loss = (net_in - x_hat) ** 2
        
        # Embedding loss
        embedding_loss = self.beta * (embedding ** 2).mean()
        
        # Total loss
        total_loss = recon_loss + embedding_loss
        
        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(embedding, x_hat)
        total_loss += grad_penalty

        if anomaly_score:
            if self.grad_flag:
                grad = torch.abs(torch.autograd.grad(total_loss.mean(), net_in)[0])
                if keepdim:
                    return torch.mean(grad, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty
                else:
                	return torch.mean(grad, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
            else:
            	if keepdim:
                    return torch.mean(total_loss, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty
            	else:
                	return torch.mean(total_loss, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
        else:
            return total_loss.mean(), recon_loss.mean(), embedding_loss, grad_penalty
            
class ConstrainedAELoss(nn.Module):
    def __init__(self, beta=0, grad_pen_weight=0,):
        super(ConstrainedAELoss, self).__init__()
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight

    def grad_pen_loss(self, embedding, y_pred):

        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        z = net_out['z']
        recon_loss = (net_in - x_hat) ** 2

	# Embedding loss
        embedding_loss = self.beta * (z ** 2).mean()

        # Total loss
        total_loss = recon_loss + embedding_loss

        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(z, x_hat)
        total_loss += grad_penalty
        
        if anomaly_score:
            return torch.mean(total_loss, dim=[1], keepdim=True), recon_loss, embedding_loss, grad_penalty if keepdim else torch.mean(total_loss, dim=[1, 2, 3]), recon_loss, embedding_loss, grad_penalty
        else:
            z_rec = net_out['z_rec']
            loss_z = (z - z_rec) ** 2
            loss = total_loss.mean() + loss_z.mean()
            return loss.mean(), total_loss.mean().item(), loss_z.mean().item(), embedding_loss


# ----------- Perceptual Loss ----------- #
class EqualLayer(nn.Module):
    def forward(self, x):
        return x


class PretrainedVGG19FeatureExtractor(nn.Module):
    def __init__(self, pad_type='zero'):
        super(PretrainedVGG19FeatureExtractor, self).__init__()
        self.pad_type = pad_type

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(1)
            padding = 0
        elif pad_type == 'zero':
            self.pad = EqualLayer()
            padding = 1
        elif pad_type == 'replication':
            self.pad = nn.ReplicationPad2d(1)
            padding = 0
        else:
            raise NotImplementedError

        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=padding)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=padding)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=padding)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=padding)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=padding)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=padding)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=padding)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=padding)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=padding)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=padding)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        for param in self.parameters():
            param.requires_grad = False

        # loading weights
        pretrained_features = vgg19(weights=VGG19_Weights.DEFAULT).features
        assert len(pretrained_features.state_dict().keys()) == len(self.state_dict().keys())

        state_dict = OrderedDict()
        for (new_name, _), (_, value) in zip(self.state_dict().items(), pretrained_features.state_dict().items()):
            state_dict[new_name] = value

        self.load_state_dict(state_dict)

    def forward(self, x, out_keys):
        out = {}

        def finished():
            return len(set(out_keys).difference(out.keys())) == 0

        out['c11'] = self.conv1_1(self.pad(x)) if not (finished()) else None
        out['r11'] = F.relu(out['c11']) if not (finished()) else None
        out['c12'] = self.conv1_2(self.pad(out['r11'])) if not (finished()) else None
        out['r12'] = F.relu(out['c12']) if not (finished()) else None
        out['p1'] = self.pool1(out['r12']) if not (finished()) else None

        out['c21'] = self.conv2_1(self.pad(out['p1'])) if not (finished()) else None
        out['r21'] = F.relu(out['c21']) if not (finished()) else None
        out['c22'] = self.conv2_2(self.pad(out['r21'])) if not (finished()) else None
        out['r22'] = F.relu(out['c22']) if not (finished()) else None
        out['p2'] = self.pool2(out['r22']) if not (finished()) else None

        out['c31'] = self.conv3_1(self.pad(out['p2'])) if not (finished()) else None
        out['r31'] = F.relu(out['c31']) if not (finished()) else None
        out['c32'] = self.conv3_2(self.pad(out['r31'])) if not (finished()) else None
        out['r32'] = F.relu(out['c32']) if not (finished()) else None
        out['c33'] = self.conv3_3(self.pad(out['r32'])) if not (finished()) else None
        out['r33'] = F.relu(out['c33']) if not (finished()) else None
        out['c34'] = self.conv3_4(self.pad(out['r33'])) if not (finished()) else None
        out['r34'] = F.relu(out['c34']) if not (finished()) else None
        out['p3'] = self.pool3(out['r34']) if not (finished()) else None

        out['c41'] = self.conv4_1(self.pad(out['p3'])) if not (finished()) else None
        out['r41'] = F.relu(out['c41']) if not (finished()) else None
        out['c42'] = self.conv4_2(self.pad(out['r41'])) if not (finished()) else None
        out['r42'] = F.relu(out['c42']) if not (finished()) else None
        out['c43'] = self.conv4_3(self.pad(out['r42'])) if not (finished()) else None
        out['r43'] = F.relu(out['c43']) if not (finished()) else None
        out['c44'] = self.conv4_4(self.pad(out['r43'])) if not (finished()) else None
        out['r44'] = F.relu(out['c44']) if not (finished()) else None
        out['p4'] = self.pool4(out['r44']) if not (finished()) else None

        out['c51'] = self.conv5_1(self.pad(out['p4'])) if not (finished()) else None
        out['r51'] = F.relu(out['c51']) if not (finished()) else None
        out['c52'] = self.conv5_2(self.pad(out['r51'])) if not (finished()) else None
        out['r52'] = F.relu(out['c52']) if not (finished()) else None
        out['c53'] = self.conv5_3(self.pad(out['r52'])) if not (finished()) else None
        out['r53'] = F.relu(out['c53']) if not (finished()) else None
        out['c54'] = self.conv5_4(self.pad(out['r53'])) if not (finished()) else None
        out['r54'] = F.relu(out['c54']) if not (finished()) else None
        out['p5'] = self.pool5(out['r54']) if not (finished()) else None
        return [out[key] for key in out_keys]


class PerceptualLoss(torch.nn.Module):
    def __init__(self,
                 reduction='mean',
                 img_weight=0,
                 # feature_weights=None,
                 feature_weights=None,
                 use_feature_normalization=False,
                 use_L1_norm=False,
                 use_relative_error=False,
                 beta=0, 
                 grad_pen_weight=0):
        super(PerceptualLoss, self).__init__()
        self.beta = beta
        self.grad_pen_weight = grad_pen_weight
        # Default value of the original paper
        if feature_weights is None:
            feature_weights = {"r42": 1}

        """
        We assume that input is normalized with 0.5 mean and 0.5 std
        """

        assert reduction in ['none', 'sum', 'mean', 'pixelwise']

        MEAN_VAR_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'vgg19_ILSVRC2012_object_detection_mean_var.pt')

        self.vgg19_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.vgg19_std = torch.Tensor([0.229, 0.224, 0.225])

        if use_feature_normalization:
            self.mean_var_dict = torch.load(MEAN_VAR_ROOT)
        else:
            self.mean_var_dict = defaultdict(
                lambda: (torch.tensor([0.0], requires_grad=False), torch.tensor([1.0], requires_grad=False))
            )

        self.reduction = reduction
        self.use_L1_norm = use_L1_norm
        self.use_relative_error = use_relative_error

        self.model = PretrainedVGG19FeatureExtractor().cuda().eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.set_new_weights(img_weight, feature_weights)

    def set_reduction(self, reduction):
        self.reduction = reduction
    def grad_pen_loss(self, embedding, y_pred):

        grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
        return torch.mean(grad ** 2)

    def forward(self, x, net_out, anomaly_score=False, keepdim=False):
        y = net_out['x_hat']
        embedding = net_out['z']
        
        
        # pixel-wise prediction is implemented only if loss is obtained from one layer of vgg
        if self.reduction == 'pixelwise':
            assert (len(self.feature_weights) + (self.img_weight != 0)) == 1

        layers = list(self.feature_weights.keys())
        weights = list(self.feature_weights.values())

        x = self._preprocess(x)
        y = self._preprocess(y)

        f_x = self.model(x, layers)
        f_y = self.model(y, layers)

        loss = None

        if self.img_weight != 0:
            loss = self.img_weight * self._loss(x, y)

        for i in range(len(f_x)):
            # put mean, var on right device
            mean, var = self.mean_var_dict[layers[i]]
            mean, var = mean.to(f_x[i].device), var.to(f_x[i].device)
            self.mean_var_dict[layers[i]] = (mean, var)

            # compute loss
            norm_f_x_val = (f_x[i] - mean) / var
            norm_f_y_val = (f_y[i] - mean) / var

            cur_loss = self._loss(norm_f_x_val, norm_f_y_val)

            if loss is None:
                loss = weights[i] * cur_loss
            else:
                loss += weights[i] * cur_loss

        loss /= (self.img_weight + sum(weights))
        
        # Embedding loss
        embedding_loss = self.beta * (embedding ** 2).mean()
        
        # Total loss
        total_loss = loss + embedding_loss
        
        grad_penalty = self.grad_pen_weight * self.grad_pen_loss(embedding, y)
        total_loss += grad_penalty
        
        if anomaly_score:
            if keepdim:
                scale_h = x.shape[-2] / loss.shape[-2]
                scale_w = x.shape[-1] / loss.shape[-1]
                total_loss = F.interpolate(total_loss, scale_factor=(scale_h, scale_w), mode='bilinear')
                return torch.mean(total_loss, dim=[1], keepdim=True), loss, embedding_loss, grad_penalty
            else:
                return torch.mean(total_loss, dim=[1, 2, 3]), loss.mean(), embedding_loss, grad_penalty

        else:
            return loss.mean(), loss.mean(), embedding_loss, grad_penalty
        # if self.reduction == 'none':
        #     return loss
        # elif self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # elif self.reduction == 'pixelwise':
        #     loss = loss.unsqueeze(1)
        #     scale_h = x.shape[2] / loss.shape[2]
        #     scale_w = x.shape[3] / loss.shape[3]
        #     loss = F.interpolate(loss, scale_factor=(scale_h, scale_w), mode='bilinear')
        #     return loss
        # else:
        #     raise NotImplementedError('Not implemented reduction: {:s}'.format(self.reduction))

    def set_new_weights(self, img_weight=0, feature_weights=None):
        self.img_weight = img_weight
        if feature_weights is None:
            self.feature_weights = OrderedDict({})
        else:
            self.feature_weights = OrderedDict(feature_weights)

    def _preprocess(self, x):
        assert len(x.shape) == 4

        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)

        # denormalize
        vector = torch.Tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1).to(x.device)
        x = x * vector + vector

        # normalize
        x = (x - self.vgg19_mean.reshape(1, 3, 1, 1).to(x.device)) / self.vgg19_std.reshape(1, 3, 1, 1).to(x.device)
        return x

    def _loss(self, x, y):
        if self.use_L1_norm:
            norm = lambda z: torch.abs(z)
        else:
            norm = lambda z: z * z

        diff = (x - y)

        if not self.use_relative_error:
            loss = norm(diff)
        else:
            means = norm(x).mean(3).mean(2).mean(1)
            means = means.detach()
            loss = norm(diff) / means.reshape((means.size(0), 1, 1, 1))

        return loss
        # # perform reduction
        # if self.reduction == 'pixelwise':
        #     return loss.mean(1)
        # else:
        #     return loss.mean(3).mean(2).mean(1)


class RelativePerceptualL1Loss(PerceptualLoss):
    def __init__(self, reduction='mean', img_weight=0, feature_weights=None, beta=0, grad_pen_weight=0):
        super().__init__(
            reduction=reduction,
            img_weight=img_weight,
            feature_weights=feature_weights,
            use_feature_normalization=True,
            use_L1_norm=True,
            use_relative_error=True,
            beta=beta,
            grad_pen_weight=grad_pen_weight)
