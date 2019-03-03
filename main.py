from __future__ import print_function
import argparse
from torch.utils.data import DataLoader
from SRCNN.solver import SRCNNTrainer
from dataset.data import get_training_set, get_test_set
from evaluate import calculate_ssim

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super resolution task')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")

args = parser.parse_args()


def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    train_set = get_training_set()
    test_set = get_test_set()
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    model = SRCNNTrainer(args, training_data_loader, testing_data_loader)

    model.run()



if __name__ == '__main__':
    main()
