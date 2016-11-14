#!/usr/bin/env python

from __future__ import print_function
import argparse
import os.path as path

from gensim.models import Word2Vec as word2vec
import numpy as np
import simplejson as json

import utils

PROJ_ROOT = path.abspath(path.join(path.dirname(__file__), '..', '..'))
DATA_ROOT = path.join(PROJ_ROOT, 'data', 'books')

# =====================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--w2v', default=path.join(DATA_ROOT, 'googlenews_w2v.bin'))
parser.add_argument('--vocab', default=path.join(DATA_ROOT, 'book_vocab.txt'))
parser.add_argument('--out-pfx', default=path.join(DATA_ROOT, 'googlenews_w2v_filt'))
args = parser.parse_args()
# =====================================================================================

w2v = word2vec.load_word2vec_format(args.w2v, binary=True)

vocab = ['</s>']
with open(args.vocab) as f_vocab:
    for word in f_vocab:
        word = word.rstrip().decode('utf8')
        if word in w2v:
            vocab.append(word)
vocab.sort(key=lambda w: -w2v.vocab[w].count)

vecs = w2v[vocab]

with open(args.out_pfx + '.bin', 'wb') as f_out, \
     open(args.out_pfx + '_vocab.txt', 'w') as f_vocab_out:
    print('%s %s' % vecs.shape, file=f_out)
    for word, vec in zip(vocab, vecs):
        word = word.encode('utf8')
        print(word, file=f_vocab_out)
        print('%s %s' % (word, vec.tostring()), file=f_out)

print('Reduced vocab size from %d to %d.' % (len(w2v.vocab), len(vocab)))
