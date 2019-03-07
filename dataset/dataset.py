from os import listdir
from os.path import join

import torch.utils.data as data
from torchvision import transforms
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    return img

# def tensor_to_PIL(tensor):
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = transforms.ToPILImage(image)
#     return image


class MyDataset_train(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(MyDataset_train, self).__init__()
        # self.image_filenames_input = [join(image_dir+"/DIV2K_train_LR_bicubic/X4", x) for x in listdir(image_dir+"/DIV2K_train_LR_bicubic/X4") if is_image_file(x)]
        self.image_filenames = [join(image_dir+"/DIV2K_train_HR", x) for x in listdir(image_dir+"/DIV2K_train_HR") if is_image_file(x)]

        # self.image_filenames_input = [join(image_dir + "/DIV2K_valid_LR_bicubic_X4/X4", x) for x in
        #                               listdir(image_dir + "/DIV2K_valid_LR_bicubic_X4/X4") if is_image_file(x)]
        # self.image_filenames_target = [
        #     join(image_dir + "/DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)", x) for x in
        #     listdir(image_dir + "/DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)") if
        #     is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)


class MyDataset_test(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(MyDataset_test, self).__init__()
        # self.image_filenames_input = [join(image_dir+"/DIV2K_valid_LR_bicubic_X4/X4", x)
        #                               for x in listdir(image_dir+"/DIV2K_valid_LR_bicubic_X4/X4") if is_image_file(x)]
        # self.image_filenames_target = [join(image_dir+"/DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)", x)
        #                                for x in listdir(image_dir+"/DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)") if is_image_file(x)]

        self.image_filenames_input = [image_dir + join("X4", x) for x in listdir(image_dir + "X4") if is_image_file(x)]
        self.image_filenames_target = [join(image_dir + "DIV2K_train_HR", x) for x in listdir(image_dir + "DIV2K_train_HR")
                                if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames_input[index])
        target = load_img(self.image_filenames_target[index])
        # target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames_input)
