import collections
import numpy as np
import os
import shutil

import torch


class StrLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self.ignore_case = ignore_case
        if self.ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for -1 index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # 0 is reserved for 'blank' required by ctc
            self.dict[char] = i + 1

    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self.ignore_case else char] for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text, _ = self.encode(''.join(text))

        return np.asarray(text, dtype=np.int32), np.asarray(length, dtype=np.int32)

    def decode(self, encode_texts, encode_probs=None, raw=False):
        texts = []
        scores = []
        for n in range(len(encode_texts)):
            encode_text = encode_texts[n]
            if raw:
                texts.append(''.join([self.alphabet[i - 1] for i in encode_text]))
            else:
                char_list = []
                min_score = 1
                for i in range(len(encode_text)):
                    if encode_text[i] != 0 and (not (i > 0 and encode_text[i - 1] == encode_text[i])):
                        char_list.append(self.alphabet[encode_text[i] - 1])
                        if encode_probs is not None:
                            char_prob = encode_probs[n, i, encode_text[i]]
                            if min_score > char_prob:
                                min_score = char_prob
                texts.append(''.join(char_list))
                scores.append(min_score)
        return texts, scores


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, checkpoint_dir, is_best=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if is_best is not None:
        checkpoint_file = '{}/checkpoint.pth.tar'.format(checkpoint_dir)
        torch.save(state, checkpoint_file)
        if is_best:
            best_checkpoint_file = '{}/checkpoint_best.pth.tar'.format(checkpoint_dir)
            shutil.copy2(checkpoint_file, best_checkpoint_file)
    else:
        checkpoint_file = '{}/checkpoint_epoch_{:04d}.pth.tar'.format(checkpoint_dir, state['epoch'])
        torch.save(state, checkpoint_file)
    return checkpoint_file


def post_checkpoint_to_eval(eval_url, val_root, checkpoint_file, max_try=10):
    import requests
    task = {
        'val_root': val_root,
        'checkpoint': checkpoint_file
    }
    count = 0
    do_while = True
    while do_while:
        try:
            response = requests.post(eval_url, json=task)
            if response.status_code == requests.codes.ok:
                print('=> Post {} to {} successful'.format(checkpoint_file, eval_url))
                break
        except Exception as e:
            count += 1
            do_while = count < max_try
            print('=> Post {} to {} error, try {} / {}'.format(checkpoint_file, eval_url, count, max_try))
