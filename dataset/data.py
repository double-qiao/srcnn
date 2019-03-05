from torchvision import transforms
from os.path import join, basename
from .dataset import MyDataset_train, MyDataset_test
def get_data_set():
    return './dataset'

def input_transform_test():

    return transforms.Compose(
        [
            Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor()

        ]
    )

def target_transform_test():

    return transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )


def input_transform_train(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
    ])


def target_transform_train(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set():
    # root_dir = get_data_set()
    # train_dir = join(root_dir, "train")
    train_dir = "/home/s1825980/datasets/DIV2K"
    # train_dir = "/home/s1825980/datasets/valid_div2k"
    crop_size = 648
    upscale_factor = 4

    return MyDataset_train(train_dir,
                             input_transform = input_transform_train(crop_size, upscale_factor),
                             target_transform = target_transform_train(crop_size))


def get_test_set():
    # root_dir = get_data_set()
    # test_dir = join(root_dir, "test")
    test_dir = "/home/s1825980/datasets/valid_div2k"
    upscale_factor = 4
    return MyDataset_test(test_dir,
                             input_transform=input_transform_test(upscale_factor),
                             target_transform=target_transform_test())