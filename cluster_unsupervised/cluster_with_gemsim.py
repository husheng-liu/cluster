#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
====================
@author:husheng.liu
@time:2020/11/12:2:28 下午
@email:liuhusheng1234@163.com
@IDE:PyCharm
====================
"""

from gensim.models.keyedvectors import KeyedVectors
import time
import numpy as np
import json
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def initial_cls(v_1, v_2, threshold, list_words):
    """

    :param v_1:
    :param v_2:
    :param threshold:
    :param list_words:
    :return:
    """
    result = np.array(model.cosine_similarities(v_1, v_2))
    # result = cosine_similarity(v_1, v_2)
    # print("result[:10]:", result[:10])
    array1 = np.where(result > threshold)[0]
    # 满足条件的向量的位置，与单词列表中的index一致。
    # print(array1)
    v_2 = np.delete(np.array(v_2), array1, axis=0)
    l5 = [list_words[num] for num in array1]
    print('相近似：', l5)
    list_words = np.delete(np.array(list_words), array1, axis=0).tolist()
    return v_2, l5, list_words


if __name__ == "__main__":
    configs = json.load(open(sys.argv[1]))
    model = KeyedVectors.load_word2vec_format(configs['v1']['path_to_vec_'], binary=False, encoding="utf-8")
    # for gensim 3.x
    # l3 = list(model.vocab.keys())
    # for gensim 4.x
    # l3 = model.index_to_key
    # # print('l3',l3)
    # l4 = model.vectors
    with open(configs['v1']['path_to_vec']) as reader:
        dict_ = json.load(reader)
    l3 = [key for key in dict_.keys()]
    l4 = [np.array(value) for value in dict_.values()]

    print('数据读取完毕,待分类单词为列表l3,'
          '共有单词{}个'.format(len(l3)))
    print("无监督聚类中")
    print("词向量阈值:", configs['v1']['threshold'])
    l6 = []
    for i in range(0, configs['v1']['num_cls']):
        # 防止不到20轮就分类完成,设置待分类列名不为空的情况下进行分类。
        if l3:
            l4, cla, l3 = initial_cls(l4[1], l4, configs['v1']['threshold'], l3)
            print('剩余待分类数量:', len(l3))
            # l6.append(" ".join(cla))
            # l7.append(len(cla))
            # if len(cla) > 50:
            #     print('第{}个类别超过{}个,输出前50个:'.format(i+1, len(cla)), cla[:50])
            # else:
            #     print('第{}个类别元素{}个,输出全部:'.format(i+1, len(cla)), cla)
            if len(cla) >= configs['v1']['num']:
                l6.append(" ".join(cla))
            else:
                print("short of {} , will delete from dictionary".format(configs['v1']['num']), cla)
                pass
        else:
            print('分类已完成,共分成{}个类别'.format(i + 1))
            break
    df = pd.DataFrame(l6, columns=['cluster_result'])
    df.to_csv(configs['v1']['save_path'], index=False, sep='\t')
    print("vec存储完毕")

    #
    # print(len(model.vectors))
    # print(len(list(model.vocab.keys())))
    # a = time.time()
    # result1 = model.distances("</s>")
    # b =time.time()
    #
    # result2 = model.cosine_similarities(model['</s>'], model.vectors)
    # c=time.time()
    # print(/-a)
    # print(c - b)
