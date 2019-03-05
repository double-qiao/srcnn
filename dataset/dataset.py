from os import listdir
from os.path import join

import torch.utils.data as data
import numpy as np
from torchvision import transforms
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath)
    return img

def resize_img(img):
    img = np.array(img)
    print(img)
    n = 4
    if img.shape[1] % n == 0 and img.shape[0] % n == 0:
        pass
    elif img.shape[1] % n == 0:
        img = img[0:(img.shape[0] - (img.shape[0] % n)), :, :]
        img = Image.fromarray(img, mode='RGB')
    elif img.shape[0] % n == 0:
        img = img[:, 0:(img.shape[1] - (img.shape[1] % n)), :]
        img = Image.fromarray(img, mode='RGB')
    else:
        img = img[0:(img.shape[0] - (img.shape[0] % n)), 0:(img.shape[1] - (img.shape[1] % n)), :]
        img = Image.fromarray(img, mode='RGB')
    # arr_img = np.asarray(img)
    # print(img.shape[1])
    # img_resize = cv2.resize(img, (int(img.shape[1]/n), int(img.shape[0]/n)), interpolation=cv2.INTER_CUBIC)
    img_resize = img.resize((int(img.size[0] / n), int(img.size[1] / n)), Image.BICUBIC)
    return img_resize

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
        target = load_img(self.image_filenames[index])
        input_image = resize_img(target)
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)


class MyDataset_test(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(MyDataset_test, self).__init__()
        # self.image_filenames_input = [join(image_dir+"/DIV2K_valid_LR_bicubic_X4/X4", x) for x in listdir(image_dir+"/DIV2K_valid_LR_bicubic_X4/X4") if is_image_file(x)]
        self.image_filenames = [join(image_dir+"/DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)", x) for x in listdir(image_dir+"/DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)") if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        input_image = resize_img(target)
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return len(self.image_filenames)