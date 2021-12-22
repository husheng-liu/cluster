#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
====================
@File:.py
@time:2021/12/21:11:32 上午
@IDE:PyCharm
====================
"""
from gensim.models import fasttext
import json
import sys
import os

if __name__ == "__main__":
    configs = json.load(open(sys.argv[1]))
    # model = FastText.load_fasttext_format(configs["path_w2v"])
    model = fasttext.FastText.load(configs["path_models"])
    print("model prepared")
    g = os.walk(configs["work_dir"])
    for path, dir_list, file_list in g:
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            print(file_path)
            f_ = open(file_path[:-4] + "-similar.txt", "w")
            with open(file_path) as f:
                for word in f.read().strip("\n").split(","):
                    if "重度" in file_path:
                        similar_word = model.similar_by_word(word, topn=20)
                        similar_word = [item[0] for item in similar_word]
                        print(similar_word)

                    else:
                        similar_word = model.similar_by_word(word, topn=10)
                        similar_word = [item[0] for item in similar_word]
                        print(similar_word)
                    f_.write(word + " :"+" ".join(similar_word)+"\n")
