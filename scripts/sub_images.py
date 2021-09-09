from random import randint
import cv2
import sys
import os
import glob

def crop_images(path, crop_w, crop_h):
    for image_path in glob.glob(path+'/*'):
        base = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        w = image.shape[0]//crop_w + 1
        h = image.shape[1]//crop_h + 1
        print(w,h)
        for idx_w in range(w):
            for idx_h in range(h):
                if (idx_w+1)*crop_w < image.shape[0] and (idx_h+1)*crop_h < image.shape[1]:
                    img = image[idx_w*crop_w:(idx_w+1)*crop_w, idx_h*crop_h:(idx_h+1)*crop_h]
                elif (idx_w+1)*crop_w >= image.shape[0] and (idx_h+1)*crop_h < image.shape[1]:
                    img = image[image.shape[0] - crop_w:image.shape[0], idx_h * crop_h:(idx_h+1)*crop_h]
                elif (idx_w+1)*crop_w < image.shape[0] and (idx_h+1)*crop_h >= image.shape[1]:
                    img = image[idx_w * crop_w:(idx_w+1)*crop_w, image.shape[1] - crop_h:image.shape[1]]
                else:
                    img = image[image.shape[0] - crop_w:image.shape[0], image.shape[1] - crop_h:image.shape[1]]
                print(idx_w, idx_h, base)
                cv2.imwrite("{:s}_sub/{:s}_{:d}.png".format(path, base, idx_w * h + idx_h), img)

def crop_images_one(path):
    if not os.path.exists("{:s}_sub".format(path)):
        os.makedirs("{:s}_crop".format(path))
    crop_w,crop_h = 4096,4096
    for image_path in glob.glob(path + '/*'):
        image = cv2.imread(image_path)
        if image.shape[0] < crop_w:
            crop_w = image.shape[0]
        if image.shape[1] < crop_h:
            crop_h = image.shape[1]
    print("crop size: ",crop_h,"*",crop_w)

    for image_path in glob.glob(path+'/*'):
        base = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        w = image.shape[0]
        h = image.shape[1]
        img = image[(w - crop_w)//2 :(w + crop_w)//2, (h - crop_h)//2:(h + crop_h)//2]
        print(base)
        cv2.imwrite("{:s}_crop/{:s}.png".format(path, base), img)

paths = "../datasets/downsampling/DIV2K100"
crop_images_one(paths)