#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: bm25.py
@time:2022/04/16
@description:
"""

import logging
import math
import os
import pickle
from typing import List, Tuple

import jieba

jieba.setLogLevel(log_level=logging.INFO)


class BM25Param(object):
    def __init__(
        self,
        f: List[dict],
        df: dict,
        idf: dict,
        length: int,
        avg_length: float,
        docs_list: List[str],
        line_length_list: List[int],
        k1: float = 1.5,
        k2: float = 1.5,
        b: float = 0.75,
    ):
        """

        :param f:
        :param df:
        :param idf:
        :param length:
        :param avg_length:
        :param docs_list:
        :param line_length_list:
        :param k1: 可调整参数，[1.2, 2.0]
        :param k2: 可调整参数，[1.2, 2.0]
        :param b:
        """
        self.f = f  # 词频矩阵,每个文档中每个词的出现次数
        self.df = df  # 文档频率,每个词出现在多少个文档中
        self.k1 = k1  # 调节因子,用于控制词频饱和度
        self.k2 = k2  # 调节因子,用于控制查询词频率的影响
        self.b = b  # 调节因子,用于控制文档长度归一化的影响
        self.idf = idf  # 逆文档频率,衡量词的重要性, idf越高,词越重要
        self.length = length  # 文档总数
        self.avg_length = avg_length  # 平均文档长度
        self.docs_list = docs_list  # 文档列表
        self.line_length_list = line_length_list  # 每个文档的长度列表

    def __str__(self):
        return f"k1:{self.k1}, k2:{self.k2}, b:{self.b}"


class BM25(object):
    _param_pkl = "data/param.pkl"
    _docs_path = "data/data.txt"
    _stop_words_path = "data/stop_words.txt"
    _stop_words = []

    def __init__(self, docs: str = ""):
        self.docs = docs
        self._stop_words = self._load_stop_words()
        self.param: BM25Param = self._load_param()

    def _load_stop_words(self) -> List[str]:
        if not os.path.exists(self._stop_words_path):
            raise Exception(f"system stop words: {self._stop_words_path} not found")
        stop_words = []
        with open(self._stop_words_path, "r", encoding="utf8") as reader:
            for line in reader:
                line = line.strip()
                stop_words.append(line)
        return stop_words

    def _build_param(self) -> BM25Param:
        def _cal_param(lines: List[str]) -> BM25Param:
            f = []
            df = {}
            length = len(lines)
            words_count = 0
            docs_list = []
            line_length_list = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                words = [
                    word
                    for word in jieba.lcut(line)
                    if word and word not in self._stop_words
                ]
                line_length_list.append(len(words))
                docs_list.append(line)
                words_count += len(words)

                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                    df[word] = df.get(word, 0) + 1
                f.append(tmp_dict)

            idf = {}
            for word, num in df.items():
                numerator = length - num + 0.5
                denominator = num + 0.5
                idf[word] = math.log(numerator) - math.log(denominator)
            """
            传统的IDF计算（log(N/n)）
            BM25算法的一个改进版本: IDF = log((length - num + 0.5) / (num + 0.5))
            N 是 length（总文档数）
            n 是 num（包含该词的文档数）
            1. 平滑处理：通过添加0.5，避免了在极端情况下（如n=0或n=N）的计算问题。
            2. 对数变换：使用对数可以压缩数值范围，使得频繁出现和罕见词之间的差异不会过大。
            3. 反比关系：随着包含该词的文档数增加，IDF值减小。这体现了"反向文档频率"的核心思想：在越多文档中出现的词，其区分度越低。
            4. 非负值：由于(N-n+0.5)总是大于(n+0.5)，所以IDF值始终为正。
            5. 范围：当词在所有文档中都出现时，IDF接近于0；当词极为罕见时，IDF接近于log(N)。
            """
            # 对IDF进行排序，得到词语重要性排名
            sorted_idf = sorted(idf.items(), key=lambda x: x[1], reverse=True)
            top_important_words = [
                word for word, _ in sorted_idf[:10]
            ]  # 获取前10个最重要的词
            print("最重要的10个词语（按IDF降序排列）:")
            for i, word in enumerate(top_important_words, 1):
                print(f"{i}. {word}: {idf[word]}")

            return BM25Param(
                f, df, idf, length, words_count / length, docs_list, line_length_list
            )

        file_path = self.docs if self.docs else self._docs_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        with open(file_path, "r", encoding="utf8") as reader:
            lines = reader.readlines()

        param = _cal_param(lines)
        with open(self._param_pkl, "wb") as writer:
            pickle.dump(param, writer)
        return param

    def _load_param(self) -> BM25Param:
        if self.docs or not os.path.exists(self._param_pkl):
            return self._build_param()

        with open(self._param_pkl, "rb") as reader:
            return pickle.load(reader)

    def _cal_similarity(self, words: List[str], index: int) -> float:
        """
        计算文档i与查询的相似度

        Args:
            words (List[str]): 查询分词后的词语列表
            index (int): 文档索引

        Returns:
            float: 文档与查询的相似度
        """
        score = 0
        for word in words:
            if word not in self.param.f[index]:
                continue
            idf = self.param.idf[word]
            f = self.param.f[index][word]
            k1, b = self.param.k1, self.param.b
            doc_len = self.param.line_length_list[index]
            avg_len = self.param.avg_length

            score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avg_len)))
            """
            这是BM25的基础版本, 公式如下:
            score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avg_len)))
            1. idf: 逆文档频率，表示词语的重要性。IDF值越高，说明该词在整个文档集中越少见，因此越重要。
            2. f: 词语在当前文档中的频率。
            3. k1: 一个可调参数，通常在1.2到2.0之间。它用来控制词频的缩放。
            4. b: 另一个可调参数，通常为0.75。它用来控制文档长度归一化的影响程度。
            5. doc_len: 当前文档的长度。
            6. avg_len: 所有文档的平均长度。
            改进版本有k2, 公式如下:
            score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * doc_len / avg_len))) * (k2 + 1) * f / (k2 + f)
            """
        return score

    def cal_similarity(self, query: str) -> List[Tuple[str, float]]:
        """
        计算查询与所有文档的相似度

        Args:
            query (str): 查询

        Returns:
            List[Tuple[str, float]]: 文档与查询的相似度
        """
        words = [
            word for word in jieba.lcut(query) if word and word not in self._stop_words
        ]

        results = []
        for i in range(self.param.length):
            doc = self.param.docs_list[i]
            similarity = self._cal_similarity(words, i)
            results.append((doc, similarity))

        return results

    def cal_similarity_rank(self, query: str) -> List[Tuple[str, float]]:
        result = self.cal_similarity(query)
        return sorted(result, key=lambda x: -x[1])


if __name__ == "__main__":
    bm25 = BM25()
    query_content = "自然语言处理并不是一般地研究自然语言"
    result = bm25.cal_similarity_rank(query_content)
    for line, score in result:
        print(line, score)
