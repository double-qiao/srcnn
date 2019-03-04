from torchvision.transforms import ToTensor
from os.path import join, basename
from .dataset import MyDataset_train, MyDataset_test
def get_data_set():
    return './dataset'

def input_transform():

    return transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

def target_transform():

    return transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

def get_training_set():
    # root_dir = get_data_set()
    # train_dir = join(root_dir, "train")
    # train_dir = "/home/s1825980/datasets/DIV2K"
    train_dir = "/home/s1825980/datasets/valid_div2k"
    return MyDataset_train(train_dir,
                             input_transform = input_transform(),
                             target_transform = target_transform())


def get_test_set():
    # root_dir = get_data_set()
    # test_dir = join(root_dir, "test")
    test_dir = "/home/s1825980/datasets/valid_div2k"
    return MyDataset_test(test_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())