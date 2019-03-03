import os
import cv2
import numpy as np
from PIL import Image


def zip(imgs_dir, imgs_ground_truth_dir, imgs_resize_dir, n):
    i = 0
    for path, subdirs, files in os.walk(imgs_dir):
        for file in files:
            i = i+1
            if i%2 == 0:
                continue
            # img = cv2.imread(imgs_dir + file)
            img = np.array(Image.open(imgs_dir + file))
            # print(img.size)
            # assert(len(img.shape)==3)
            # assert(img.shape[2] == 3)
            # print(img.shape[1])
            if img.shape[1]%n == 0 and img.shape[0]%n == 0:
                # cv2.imwrite(imgs_ground_truth_dir + file, img)
                img = Image.fromarray(img, mode='RGB')
                img.save(imgs_ground_truth_dir + file)
            elif img.shape[1]%n == 0:
                img = img[0:(img.shape[0]-(img.shape[0]%n)), :, :]
                # cv2.imwrite(imgs_ground_truth_dir + file, img)
                img = Image.fromarray(img, mode='RGB')
                img.save(imgs_ground_truth_dir + file)
            elif img.shape[0]%n == 0:
                img = img[:, 0:(img.shape[1] - (img.shape[1]%n)), :]
                # cv2.imwrite(imgs_ground_truth_dir + file, img)
                img = Image.fromarray(img, mode='RGB')
                img.save(imgs_ground_truth_dir + file)
            else:
                img = img[0:(img.shape[0] - (img.shape[0]%n)), 0:(img.shape[1] - (img.shape[1]%n)), :]
                # cv2.imwrite(imgs_ground_truth_dir + file, img)
                img = Image.fromarray(img, mode='RGB')
                img.save(imgs_ground_truth_dir + file)
            # arr_img = np.asarray(img)
            # print(img.shape[1])
            # img_resize = cv2.resize(img, (int(img.shape[1]/n), int(img.shape[0]/n)), interpolation=cv2.INTER_CUBIC)
            img_resize = img.resize((int(img.size[0]/n), int(img.size[1]/n)), Image.BICUBIC)
            # cv2.imwrite(imgs_resize_dir + file, img_resize)
            img_resize.save(imgs_resize_dir + file)

def zip1(imgs_dir, imgs_resize_dir, n):
    i = 0
    for path, subdirs, files in os.walk(imgs_dir):
        for file in files:
            # i = i+1
            # if i%2 == 0:
            #     continue
            # img = cv2.imread(imgs_dir + file)
            img = Image.open(imgs_dir + file)
            # assert(len(img.shape)==3)
            # assert(img.shape[2] == 3)
            # print(img.shape[1])
            # if img.shape[1]%n == 0 and img.shape[0]%n == 0:
            #     cv2.imwrite(imgs_ground_truth_dir + file, img)
            # elif img.shape[1]%n == 0:
            #     img = img[0:(img.shape[0]-(img.shape[0]%n)), :, :]
            #     cv2.imwrite(imgs_ground_truth_dir + file, img)
            # elif img.shape[0]%n == 0:
            #     img = img[:, 0:(img.shape[1] - (img.shape[1]%n)), :]
            #     cv2.imwrite(imgs_ground_truth_dir + file, img)
            # else:
            #     img = img[0:(img.shape[0] - (img.shape[0]%n)), 0:(img.shape[1] - (img.shape[1]%n)), :]
            #     cv2.imwrite(imgs_ground_truth_dir + file, img)
            # arr_img = np.asarray(img)
            # print(img.shape[1])
            # img_resize = cv2.resize(img, (int(img.shape[1]/n), int(img.shape[0]/n)), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(imgs_resize_dir + file, img_resize)
            img_resize = img.resize((int(img.size[0] / n), int(img.size[1] / n)), Image.BICUBIC)
            img_resize.save(imgs_resize_dir + file)


# imgs_dir = "./DIV2K_valid_HR/"
# imgs_ground_truth_dir = "./DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)/"
# imgs_resize_dir = "./DIV2K_valid_LR_bicubic_X16/X16_valid/"
n=4
# imgs_dir = "./DIV2K_train_HR/"
# imgs_ground_truth_dir = "./DIV2K_train_HR(srgan for X16)/DIV2K_train_HR_ground_truth(srgan for X16)/"
# imgs_resize_dir = "./DIV2K_train_LR_bicubic_X16/X16_vaild/"
# imgs_dir = "./DIV2K_valid_HR(srgan for X16)/DIV2K_valid_HR_ground_truth(srgan for X16)/"
# imgs_resize_dir = "./DIV2K_valid_LR_bicubic_X4/X4/"

imgs_dir = "./CAT_00(srgan for X16)/CAT_00_ground_truth(srgan for X16)/"
imgs_resize_dir = "./CAT_00_X4/X4_CAT/"


zip1(imgs_dir,imgs_resize_dir, n)
