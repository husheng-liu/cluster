## 步骤
### 1 首先对原始数据进行无监督聚类，得到聚类的词与词向量后，然后依据词的先后顺序依次进行cosine_similarities计算，设置阈值，从而获得第一轮聚类结果。如果不符合预期结果，或者差距较大，可以进行第二轮聚类。
### 2 得到聚类后的词与词向量后，可以删掉高频但无意义的词语。防止高频词汇无意义词汇与有效词汇聚类。
### 3 todo：
     可以通过gensim 相关api 对cosine_similarities计算进行优化