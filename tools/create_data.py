import argparse
import os
import lmdb
import cv2
import json
import numpy as np


def check_image_is_valid(image_buf):
    try:
        image = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
    except Exception:
        return False
    im_h, im_w, _ = image.shape
    if im_h * im_w == 0:
        return False
    return True


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def create_dataset(save_root, meta_info, image_list, label_list):
    assert len(image_list) == len(label_list)

    env = lmdb.open(save_root, map_size=1099511627776)

    count = 0
    cache = {}

    for path, label in zip(image_list, label_list):
        if not os.path.exists(path):
            print('{} does not exist'.format(path))
            continue

        with open(path, 'rb') as f:
            image_buf = np.frombuffer(f.read(), dtype=np.uint8)

        if not check_image_is_valid(image_buf):
            print('{} is not a valid image'.format(path))
            continue

        cache['{}{:08d}'.format(meta_info['image_key_prefix'], count).encode()] = image_buf.tobytes()
        cache['{}{:08d}'.format(meta_info['label_key_prefix'], count).encode()] = label.encode()

        if count % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written {} / {}'.format(count, len(image_list)))

        count += 1

    meta_info['num_samples'] = count
    cache['meta_info'.encode()] = json.dumps(meta_info).encode()
    write_cache(env, cache)

    print('Created dataset with {} samples'.format(count))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', type=str, default='/workspace/mnt/group/video/luanjun/datasets/ocr/plate/list/crop_plate.txt')
    parser.add_argument('--image_root', type=str, default='/workspace/mnt/group/video/luanjun/datasets/ocr/plate')
    parser.add_argument('--save_root', type=str, default='/workspace/mnt/group/video/luanjun/datasets/ocr/plate/lmdb/crop_plate_lmdb_train')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    image_list = []
    label_list = []
    with open(args.list_file, 'r') as f:
        for line in f:
            path, label = line.strip().split(' ')
            image_list.append('{}/{}'.format(args.image_root, path))
            if '#' in label:
                label = label.replace('#', '')
            label_list.append(label)

    meta_info = {
        'num_samples': 0,
        'image_key_prefix': 'image/',
        'label_key_prefix': 'label/'
    }

    create_dataset(args.save_root, meta_info, image_list, label_list)
