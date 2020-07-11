#!/usr/bin/env python


"""Check if all the image have extracted features and also add movie_prefix to VCR dataset """

# Example:
# python ./tools/generate_tsv_vcr.py --ann_file train.jsonl --data_root {VCR_Root} --out {VCR_Root}/train_frcnn/


import _init_paths
from utils.timer import Timer
from zip_helper import ZipHelper

import argparse
import pprint
import time, os, sys
import numpy as np
import json
import jsonlines

def load_image_ids(ann_file, data_root, outfolder):
    tic = time.time()
    if ann_file == 'train.jsonl':
        prefix = 'train'
    elif ann_file == 'val.jsonl':
        prefix = 'val'
    else:
        raise Exception('Unknown annotation file')
    ann_file = os.path.join(data_root, ann_file)
    split = []
    frcnn = []
    # ignore or not find cached database, reload it from annotation file
    print('loading database from {}...'.format(ann_file))

    with jsonlines.open(ann_file) as reader:
        for progress, ann in enumerate(reader):
            img_fn = os.path.join(data_root, 'vcr1images.zip@/vcr1images', ann['img_fn'])
            movie_id = img_fn.split('/')[-2]
            image_id = img_fn.split('/')[-1][:-4]
            if os.path.exists(os.path.join(outfolder, '{}.json'.format(image_id))):
                src_path = os.path.join(outfolder, '{}.json'.format(image_id))
                if os.stat(src_path).st_size == 0:
                    print('Not exist {}'.format(image_id))
                    os.system("mv {} /tmp/trash".format(src_path))
                else:
                    tgt_path =os.path.join(outfolder, '{}.json'.format(movie_id + '_' + image_id))  
                    os.system('mv "{}" "{}"'.format(src_path, tgt_path))
            image_id = movie_id + '_' + image_id
            if progress % 1000 == 0:
                print(" Progress: {}".format(progress))
    print('Done (t={:.2f}s)'.format(time.time() - tic))
    return split


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--out', dest='outfolder',
                        help='output folder',
                        default=None, type=str)
    parser.add_argument('--data_root', dest='data_root',
                        help='data root path',
                        default=None, type=str)
    parser.add_argument('--ann_file', dest='ann_file',
                        help='the annotation file from vcr', 
                        default='train.jsonl', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

    

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)
    image_ids = load_image_ids(args.ann_file, args.data_root, args.outfolder)


