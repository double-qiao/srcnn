from __future__ import print_function
from torchvision import transforms
# from math import log10

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from SRCNN.model import Net
from progressbar import *
from SRCNN.evaluate import calculate_ssim, calculate_psnr
from PIL import Image





class SRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(SRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.train_batchsize = config.batchSize
        self.test_batchsize = config.testBatchSize

    def build_model(self):
        self.model = Net(num_channels=3, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def save_model(self):
        model_out_path = "model_path.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def tensor_to_PIL(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        return image

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            ProgressBar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / (len(self.training_loader) / self.train_batchsize)))

    def test(self):
        self.model.eval()
        avg_psnr = 0
        avg_ssim = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                for i in range(self.test_batchsize):
                    img = prediction[i, :, :, :]
                    Img = self.tensor_to_PIL(img)


                    # img = img.cpu().numpy()
                    # img_arr = np.transpose(img, (1, 2, 0))
                    # print(img_arr.shape)
                    # Img = Image.fromarray(img_arr, mode='RGB')
                    string = str((self.test_batchsize*batch_num)+i)
                    Img.save("/home/s1825980/srcnn/SRCNN/predict/" + string +'.jpg')
                ssim = calculate_ssim(prediction, target)
                # mse = self.criterion(prediction, target)
                # psnr = 10 * log10(1 / mse.item())
                psnr = calculate_psnr(prediction, target)
                avg_psnr += psnr
                avg_ssim += ssim
                ProgressBar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)), 'SSIM: %.4f' % (avg_ssim / (batch_num + 1)))

        print(" ----Average PSNR/SSIM results for ----\n\tPSNR: {:.4f} dB; SSIM: {:.4f}\n".format(avg_psnr / (len(self.testing_loader)/self.test_batchsize), avg_ssim / (len(self.testing_loader)/self.test_batchsize)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()  # validate
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save_model()


    def run_test(self):
        self.model = torch.load('model_path.pth')
        self.model.eval()
        self.test()