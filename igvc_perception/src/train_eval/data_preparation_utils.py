# import the necessary packages
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob, os
import send2trash
import argparse
 
def compare_images(imageA, imageB):
	# compute the structural similarity
    s = measure.compare_ssim(imageA, imageB)
    return s

def delete_file(delete_list):
    # Delete files
    for delete_file in delete_list:
        print("file is deleted: " + delete_file)
        send2trash.send2trash(delete_file)

def image_processing(imgs_folder,image_path):
    # Convert images into gray scale
    path = os.path.join(imgs_folder, image_path)
    image = cv2.imread(path)
    image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return path, image_converted

def find_similar_images(imgs_folder):
    # Initialize delete list
    delete_list = []
    sorted_folder_dir = sorted(os.listdir(imgs_folder))
    # Start comparing two images at a time
    for imageA, imageB in zip(sorted_folder_dir[::],sorted_folder_dir[1::]):
        print("comparing: " + imageA,imageB)
        pathA, imageA = image_processing(imgs_folder, imageA)
        pathB, imageB = image_processing(imgs_folder, imageB)
        result = compare_images(imageA, imageB)
        if result >= 0.8:
            delete_list.append(pathA)
    return delete_list

def check_how_many_folders(folder_dir):
    folder_list = []
    for root, dirs, files in os.walk(folder_dir, topdown=False):
        for name in dirs:
            folder_list.append(os.path.join(root, name))
    
    return folder_list

def create_test_train_files(dataset_path,file_train,file_test):
    # Percentage of images to be used for the test set
    percentage_test = 10

    # Populate train.txt and test.txt
    counter = 1
    index_test = round(100 / percentage_test)
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.png")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test+1:
            counter = 1
            file_test.write(dataset_path + "/" + title + '.png' + "\n")
        else:
            file_train.write(dataset_path + "/" + title + '.png' + "\n")
            counter = counter + 1

def main():
    print("Data Preparation Tool for IGVC")
    parser = argparse.ArgumentParser(description='Compare images and remove repeated images.')
    parser.add_argument('imgs_folder', type=str, default=None,
                help='Directory containing images.')
    parser.add_argument('--remove_image', action='store_true', default=False,
                        help='Remove repeated images after extracting raw images from bag files')
    parser.add_argument('--test_train_generation', action='store_true', default=False,
                        help='Create a test and train file for neuronet training.')
    parser.add_argument('--reduce_data_size', type=int, default=2,
                help='Type your factor to devide your size of image folder')
    args = parser.parse_args()

    if args.remove_image:
        folder_list = check_how_many_folders(args.imgs_folder)
        
        if len(folder_list) == 0:
            print("You have " + str(len(folder_list)) + " folders")
            delete_list = find_similar_images(args.imgs_folder)
            delete_file(delete_list)
            print("Finished cleaning up files")
        else:
            print("You have " + str(len(folder_list)) + " folders")
            for directory in folder_list:
                delete_list = find_similar_images(directory)
                delete_file(delete_list)
                print("Finished cleaning up files")

    if args.test_train_generation:
        folder_list = check_how_many_folders(args.imgs_folder)
        # Create and/or truncate train.txt and test.txt
        file_train = open('train.txt', 'w')
        file_test = open('test.txt', 'w')
        print("train and test files are saved here: " + str(os.getcwd()))
        for directory in folder_list:
            create_test_train_files(directory,file_train,file_test)
    
    if args.reduce_data_size:
        folder_list = check_how_many_folders(args.imgs_folder)
        print("Your data size is: " + str(len(os.listdir(args.imgs_folder))))
        for index, image in enumerate(sorted(os.listdir(args.imgs_folder))):
            if index % args.reduce_data_size == 0:
                file_path = os.path.join(args.imgs_folder, image)
                send2trash.send2trash(file_path)
        print("Your data size now became: " + str(len(os.listdir(args.imgs_folder))))


if __name__ == '__main__':
    main()