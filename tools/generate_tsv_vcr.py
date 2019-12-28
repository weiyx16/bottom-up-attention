#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# python ./tools/generate_tsv_vcr.py --gpu 0,1,2,3 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --ann_file train.jsonl --data_root {VCR_Root} --out {VCR_Root}/train_frcnn/


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from zip_helper import ZipHelper

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json
import jsonlines

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'classes', 'attrs', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 10
MAX_BOXES = 36

def load_image_ids(ann_file, data_root):
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
        for ann in reader:
            img_fn = os.path.join(data_root, 'vcr1images.zip@/vcr1images', ann['img_fn'])
            image_id = img_fn.split('/')[-1][:-4]
            split.append((img_fn,image_id))
            frcnn.append({'image':"vcr1images.zip@/vcr1images/{}".format(ann['img_fn']), 'frcnn':prefix+"_frcnn.zip@/{}.json".format(ann['img_fn'])})
    
    if not os.path.exists(os.path.join(data_root, prefix+'_frcnn.json')):
        with open(os.path.join(data_root, prefix+'_frcnn.json'), 'w') as outfile:
            for frcnn_detail in frcnn:
                json.dump(frcnn_detail, outfile)
                outfile.write('\n')

    print('Done (t={:.2f}s)'.format(time.time() - tic))
    return split

    
def get_detections_from_im(net, im_file, image_id, ziphelper, data_root, conf_thresh=0.5):
    print im_file
    zip_image = ziphelper.imread(str(os.path.join(data_root, im_file)))
    im = cv2.cvtColor(np.array(zip_image), cv2.COLOR_RGB2BGR)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
   
    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'classes': base64.b64encode(scores[keep_boxes]),
        'attrs': base64.b64encode(attr_scores[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes])
    }   


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfolder',
                        help='output folder',
                        default=None, type=str)
    parser.add_argument('--data_root', dest='data_root',
                        help='data root path',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--ann_file', dest='ann_file',
                        help='the annotation file from vcr', 
                        default='train.jsonl', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

    
def generate_tsv(gpu_id, prototxt, weights, image_ids, data_root, outfolder):
    # First check if file exists, and if it is complete
    wanted_ids = set([image_id[1] for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfolder):
        for ids in wanted_ids:
            json_file = "{}.json".format(ids)
            if os.path.exists(os.path.join(outfolder, json_file)):
                found_ids.add(ids)

    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print 'GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids))
    else:
        print 'GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(wanted_ids))

    worked_ids = set()
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        ziphelper = ZipHelper()
        _t = {'misc' : Timer()}
        count = 0
        for im_file,image_id in image_ids:
            if image_id in missing and image_id not in worked_ids:
                worked_ids.add(image_id)
                _t['misc'].tic()
                json_file = "{}.json".format(image_id)
                with open(os.path.join(outfolder, json_file), 'w') as f:
                    json.dump(get_detections_from_im(net, im_file, image_id, ziphelper, data_root), f)
                _t['misc'].toc()
                if (count % 100) == 0:
                    print '\r\n GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours) \r\n' \
                          .format(gpu_id, count+1, len(missing), _t['misc'].average_time, 
                          _t['misc'].average_time*(len(missing)-count)/3600)
                count += 1

     
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(args.ann_file, args.data_root)
    random.seed(10)
    random.shuffle(image_ids)
    # Split image ids between gpus
    if not os.path.exists(args.outfolder):
      os.makedirs(args.outfolder)

    image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]
    
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []    
    
    for i,gpu_id in enumerate(gpus):
        p = Process(target=generate_tsv,
                    args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], args.data_root, args.outfolder))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()            
                  
