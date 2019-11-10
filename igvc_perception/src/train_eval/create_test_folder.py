#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys
import shutil
import numpy as np
import labelme

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('imgs_folder', help='input imgs folder')
    parser.add_argument('annotation', help='input annotation file')
    parser.add_argument('test_text', help='input test file')
    parser.add_argument('output_dir', help='output dataset directory')
    args = parser.parse_args()
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'imgs'))
    os.makedirs(osp.join(args.output_dir, 'annos'))
    print('Creating dataset to', args.output_dir)
    for line in open(args.test_text).readlines():
        os.chdir(args.imgs_folder)
        os.listdir(args.imgs_folder)
        n = len(line.split("/")) - 1
        test = line.split("/")[n].strip().replace("\n", "")
        if test in os.listdir(args.imgs_folder):
            shutil.copy(test, osp.join(args.output_dir, 'imgs'))
        os.chdir(args.annotation)
        os.listdir(args.annotation)
        n = len(line.split("/")) - 1
        test = line.split("/")[n].strip().replace("\n", "")
        test = test.replace(".jpg", ".png")
        print(test)
        if test in os.listdir(args.annotation):
            shutil.copy(test, osp.join(args.output_dir, 'annos'))
    
if __name__ == '__main__':
    main()
