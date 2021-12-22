#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
====================
@File:.py
@time:2021/10/21:12:02 下午
@IDE:PyCharm
====================
"""

from gensim.models import FastText, word2vec

import json
import sys


if __name__ == '__main__':
    configs = json.load(open(sys.argv[1]))

    sentences = word2vec.LineSentence(open(configs['path_raw_after_token']))

    model = FastText(sentences=sentences, min_count=5, sg=1, size=256, window=5, workers=10, min_n=2, max_n=5, word_ngrams=0, iter=10)
    # model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=model.corpus_count, epochs=model.iter, total_words=model.corpus_total_words)
    # model.wv.save_word2vec_format(configs['path_w2v'], configs["path_vocab"], binary=True)
    model.save(configs["models"])




