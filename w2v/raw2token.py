#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
====================
@File:.py
@time:2021/10/21:3:04 下午
@IDE:PyCharm
====================
"""

from tokenizers import BertWordPieceTokenizer
import sys
import json


class BertTokenizer:
    def __init__(self, path_vocab):
        self.BertTokenizer = BertWordPieceTokenizer(path_vocab)

    def encode(self, text: str):
        tokens_ = self.BertTokenizer.encode(text).tokens
        # tok = tokens_
        return tokens_


if __name__ == "__main__":
    configs = json.load(open(sys.argv[1]))
    tokenizer = BertTokenizer("en_bert.txt")
    f_w = open(configs['path_raw_after_token'], 'w')
    with open(configs["path_raw_file"], 'r') as f:
        for seq in f:
            tokens = tokenizer.encode(seq.strip('\n'))
            print(tokens)
            f_w.write(" ".join(tokens)+'\n')
