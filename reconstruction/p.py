import torch
import torch.nn as nn
import torch.autograd
from networks.base_units.blocks import BasicBlock, BottleNeck, SpatialBottleNeck
class AELoss(nn.Module):
    def __init__(self, beta=1.0, apply_grad_pen=False, grad_pen_weight=1.0, entropy_qz=None, regularization_loss=None):
        super(AELoss, self).__init__()
        self.beta = beta
        self.apply_grad_pen = apply_grad_pen
        self.grad_pen_weight = grad_pen_weight
        self.entropy_qz = entropy_qz
        self.regularization_loss = regularization_loss

    def per_pix_recon_loss(self, y_true, y_pred):
        return torch.mean(torch.sum((y_true - y_pred) ** 2, dim=[1, 2, 3]))

    def embeddig_loss(self, embedding):
        return torch.mean(torch.sum(embedding ** 2, dim=1))

    def grad_pen_loss(self, embedding, y_pred):
        # Ensure embedding requires grad
        if not embedding.requires_grad:
            print("Embedding does not require grad.")
            embedding.requires_grad = True

        print(f"Embedding requires grad: {embedding.requires_grad}")

        if self.entropy_qz is not None:
            grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
            return torch.mean((self.entropy_qz * grad) ** 2)
        else:
            grad = torch.autograd.grad(outputs=(y_pred ** 2).mean(), inputs=embedding, create_graph=True, retain_graph=True)[0]
            return torch.mean(grad ** 2)

    def forward(self, net_in, net_out, anomaly_score=False, keepdim=False):
        x_hat = net_out['x_hat']
        embedding = net_out['z']

        # Print embedding requires_grad
        print(f"Before requires_grad_: Embedding requires grad: {embedding.requires_grad}")
        
        # Ensure embedding requires grad during forward pass
        embedding = embedding.requires_grad_()

        # Print embedding requires_grad after setting it
        print(f"After requires_grad_: Embedding requires grad: {embedding.requires_grad}")

        # Reconstruction loss
        recon_loss = self.per_pix_recon_loss(net_in, x_hat)
        
        # Embedding loss
        embedding_loss = self.beta * self.embeddig_loss(embedding)
        
        # Total loss
        total_loss = recon_loss + embedding_loss
        
        # Gradient penalty loss (if applicable)
        if self.apply_grad_pen:
            grad_penalty = self.grad_pen_weight * self.grad_pen_loss(embedding, x_hat)
            total_loss += grad_penalty

        # Additional regularization loss (if any)
        if self.regularization_loss is not None:
            total_loss += self.regularization_loss

        if anomaly_score:
            if self.grad_score:
                grad = torch.abs(torch.autograd.grad(recon_loss.mean(), net_in, create_graph=True, retain_graph=True)[0])
                return torch.mean(grad, dim=[1], keepdim=keepdim) if keepdim else torch.mean(grad, dim=[1, 2, 3])
            else:
                return torch.mean(recon_loss, dim=[1], keepdim=keepdim) if keepdim else torch.mean(recon_loss, dim=[1, 2, 3])
        else:
            return total_loss

class AE(nn.Module):
    def __init__(self, input_size=64, in_planes=1, base_width=16, expansion=1, mid_num=2048, latent_size=16,
                 en_num_layers=1, de_num_layers=1, spatial=False):
        super(AE, self).__init__()

        bottleneck = SpatialBottleNeck if spatial else BottleNeck

        self.fm = input_size // 16  # down-sample for 4 times. 2^4=16

        self.en_block1 = BasicBlock(in_planes, 1 * base_width * expansion, en_num_layers, downsample=True)
        self.en_block2 = BasicBlock(1 * base_width * expansion, 2 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block3 = BasicBlock(2 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)
        self.en_block4 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, en_num_layers,
                                    downsample=True)

        self.bottle_neck = bottleneck(4 * base_width * expansion, feature_size=self.fm, mid_num=mid_num,
                                      latent_size=latent_size)

        self.de_block1 = BasicBlock(4 * base_width * expansion, 4 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block2 = BasicBlock(4 * base_width * expansion, 2 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block3 = BasicBlock(2 * base_width * expansion, 1 * base_width * expansion, de_num_layers,
                                    upsample=True)
        self.de_block4 = BasicBlock(1 * base_width * expansion, in_planes, de_num_layers, upsample=True,
                                    last_layer=True)

    def forward(self, x):
        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        bottle_out = self.bottle_neck(en4)
        z, de4 = bottle_out['z'], bottle_out['out']

        de3 = self.de_block1(de4)
        de2 = self.de_block2(de3)
        de1 = self.de_block3(de2)
        x_hat = self.de_block4(de1)

        # Ensure embedding z requires grad
        z = z.requires_grad_()

        return {'x_hat': x_hat, 'z': z, 'en_features': [en1, en2, en3], 'de_features': [de1, de2, de3]}

# Ensure model parameters require gradients
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

# Temporary main function for debugging
def main():
    model = AE()
    set_requires_grad(model, requires_grad=True)
    criterion = AELoss(beta=1.0, apply_grad_pen=True, grad_pen_weight=1.0)

    # Dummy data for testing
    img = torch.randn(1, 1, 64, 64, requires_grad=True)
    net_out = model(img)
    loss = criterion(img, net_out)

    # Backward pass to check gradients
    loss.backward()
    print("Loss computed and backward pass completed.")

if __name__ == "__main__":
    main()

