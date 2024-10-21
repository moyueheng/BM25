#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: bm25.py
@time:2022/04/16
@description:
"""
import math
import os
import jieba
import pickle
import logging
import numpy as np
from typing import List, Tuple

jieba.setLogLevel(log_level=logging.INFO)


class BM25Param(object):
    def __init__(self, f: List[dict], df: dict, idf: dict, length: int, avg_length: float, 
                 docs_list: List[str], line_length_list: List[int], k1: float = 1.5, k2: float = 1.0, b: float = 0.75):
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
        self.f = f
        self.df = df
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.idf = idf
        self.length = length
        self.avg_length = avg_length
        self.docs_list = docs_list
        self.line_length_list = line_length_list

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
        with open(self._stop_words_path, 'r', encoding='utf8') as reader:
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
                words = [word for word in jieba.lcut(line) if word and word not in self._stop_words]
                line_length_list.append(len(words))
                docs_list.append(line)
                words_count += len(words)
                
                tmp_dict = {}
                for word in words:
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                    df[word] = df.get(word, 0) + 1
                f.append(tmp_dict)

            idf = {word: math.log(length - num + 0.5) - math.log(num + 0.5) for word, num in df.items()}
            return BM25Param(f, df, idf, length, words_count / length, docs_list, line_length_list)

        file_path = self.docs if self.docs else self._docs_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")

        with open(file_path, 'r', encoding='utf8') as reader:
            lines = reader.readlines()
        
        param = _cal_param(lines)
        with open(self._param_pkl, 'wb') as writer:
            pickle.dump(param, writer)
        return param

    def _load_param(self) -> BM25Param:
        if self.docs or not os.path.exists(self._param_pkl):
            return self._build_param()
        
        with open(self._param_pkl, 'rb') as reader:
            return pickle.load(reader)

    def _cal_similarity(self, words: List[str], index: int) -> float:
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
        return score

    def cal_similarity(self, query: str) -> List[Tuple[str, float]]:
        words = [word for word in jieba.lcut(query) if word and word not in self._stop_words]
        return [(self.param.docs_list[i], self._cal_similarity(words, i)) for i in range(self.param.length)]

    def cal_similarity_rank(self, query: str) -> List[Tuple[str, float]]:
        result = self.cal_similarity(query)
        return sorted(result, key=lambda x: -x[1])



if __name__ == '__main__':
    bm25 = BM25()
    query_content = "自然语言处理并不是一般地研究自然语言"
    result = bm25.cal_similarity(query_content)
    for line, score in result:
        print(line, score)
    print("**"*20)
    result = bm25.cal_similarity_rank(query_content)
    for line, score in result:
        print(line, score)
