import time

from utils.ae_worker import AEWorker
from utils.util import AverageMeter


class MemAEWorker(AEWorker):
    def __init__(self, opt):
        super(MemAEWorker, self).__init__(opt)

    def train_epoch(self):
        self.net.train()
        losses, recon_losses, entro_losses = AverageMeter(), AverageMeter(), AverageMeter()
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()

            net_out = self.net(img)

            loss, recon_loss, entro_loss, _, _ = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = img.size(0)
            losses.update(loss.item(), bs)
            recon_losses.update(recon_loss, bs)
            entro_losses.update(entro_loss, bs)
        return losses.avg, recon_losses.avg, entro_losses.avg

    def run_train(self):
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, recon_loss, entro_loss = self.train_epoch()
            self.logger.log(step=epoch, data={"train/loss": train_loss,
                                              "train/recon_loss": recon_loss,
                                              "train/entro_loss": entro_loss})

            if epoch == 1 or epoch % self.opt.train['eval_freq'] == 0:
                eval_results = self.evaluate()

                t = time.time() - t0
                print("Epoch[{:3d}/{:3d}]  Time:{:.1f}s  loss:{:.5f}  recon_loss:{:.5f}  entro_loss:{:.5f}".format(
                    epoch, num_epochs, t, train_loss, recon_loss, entro_loss), end="  |  ")

                keys = list(eval_results.keys())
                for key in keys:
                    print(key+": {:.4f}".format(eval_results[key]), end="  ")
                    eval_results["val/"+key] = eval_results.pop(key)
                print()

                self.logger.log(step=epoch, data=eval_results)
                t0 = time.time()

        self.save_checkpoint()
        self.logger.finish()
