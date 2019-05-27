import cv2
import numpy as np
import os
from send2trash import send2trash

imgs_path = "/home/yyajima/Desktop/annotation_files_367_550_alejandro"
img_folder = []
annos_folder = []

def create_imgs_folder(imgs_path):
    for img in os.listdir(imgs_path):
        if img.endswith('.jpg'):
            img_folder.append(img)

def check_same_imgs(img_folder):
    for first_img in img_folder:
        first_img_path = os.path.join(imgs_path, first_img)
        for second_img in img_folder:
            second_img_path = os.path.join(imgs_path, second_img)
            if second_img is not first_img:
                first_img_array = cv2.imread(first_img_path)
                second_img_array = cv2.imread(second_img_path)
                difference = cv2.subtract(first_img_array,second_img_array)
                result = not np.any(difference)
                if result is True:
                    send2trash(second_img_path)
                    print("The images are the same")

def main(imgs_path):
    create_imgs_folder(imgs_path)
    check_same_imgs(img_folder)

if __name__ == '__main__':
    main(imgs_path)