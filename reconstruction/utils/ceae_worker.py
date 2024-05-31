from utils.ae_worker import AEWorker
from utils.util import AverageMeter


class CeAEWorker(AEWorker):
    def __init__(self, opt):
        super(CeAEWorker, self).__init__(opt)

    def train_epoch(self):
        self.net.train()
        losses, recon_losses, embedding_losses, grad_losses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img, img_masked = data_batch['img'], data_batch['img_masked']

            img, img_masked = img.cuda(), img_masked.cuda()

            net_out = self.net(img_masked)

            loss, recon_loss, embedding_loss, grad_loss = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = img.size(0)
            losses.update(loss.item(), bs)
            recon_losses.update(recon_loss.item(), bs)
            embedding_losses.update(embedding_loss.item(), bs)
            grad_losses.update(grad_loss.item(), bs)
        return losses.avg, recon_losses.avg, embedding_losses.avg, grad_losses.avg
