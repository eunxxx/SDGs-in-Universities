from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import normalize

import os
import pandas as pd
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma

def scan_vocabulary(sents, tokenize, min_count=2):
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w:c for w,c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx


def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2):
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k:v for k,v in counter.items() if v >= min_cooccurrence}
    n_vocabs = len(vocab_to_idx)
    return dict_to_mat(counter, n_vocabs, n_vocabs)

def dict_to_mat(d, n_rows, n_cols):
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

def word_graph(sents, tokenize=None, min_count=2, window=2, min_cooccurrence=2):
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    tokens = [tokenize(sent) for sent in sents]
    g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence) # verbose 제거
    return g, idx_to_vocab

def pagerank(x, df=0.85, max_iter=30):
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1,1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1,1)

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R

def textrank_keyword(sents, tokenize, min_count, window, min_cooccurrence, df=0.85, max_iter=30, topk=30):
    g, idx_to_vocab = word_graph(sents, tokenize, min_count, window, min_cooccurrence)
    R = pagerank(g, df, max_iter).reshape(-1)
    idxs = R.argsort()[-topk:]
    keywords = [(idx_to_vocab[idx], R[idx]) for idx in reversed(idxs)]
    return keywords

def df_to_kwdf(df, category_num):
    class_ = pd.DataFrame(df[df['5p']==category_num]) # 데이터프레임에 따라 칼럼명 변경하기
    class_list = [text for text in class_['내용_전처리']] # 데이터프레임에 따라 칼럼명 변경하기
    
    class_mecab = list(textrank_keyword(class_list, tokenize=mecab_tokenizer, min_count=2, window=2, min_cooccurrence=1))
    
    kw_list = []
    
    for i in range(len(class_mecab)):
        kw_str = class_mecab[i][0].split('/')[0]
        kw_pos = class_mecab[i][0].split('/')[1]
        kw_rank = class_mecab[i][1]
        list_ = [kw_str, kw_pos, kw_rank]
        kw_list.append(list_)

    kwdf = pd.DataFrame(kw_list, columns=['키워드', '품사', 'textrank'])

    return kwdf

def keyword_extraction(config):
    
    path = config.load_path
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith('.csv')]

    for i in file_list_py:
        globals()['data'+i.split('.')[0][-1]] = pd.read_csv(path + i)

    kw_list_all = []
    data_list = [data1, data2, data3, data4, data5]

    for data in data_list:
        for kw in data['키워드']:
            kw_list_all.append(kw)
    
    elem = [] # 처음 등장한 값인지 판별하는 리스트
    dup = [] # 중복된 원소만 넣는 리스트

    for i in kw_list_all:
        if i not in elem: # 처음 등장한 원소
            elem.append(i)
        else:
            if i not in dup: # 이미 중복 원소로 판정된 경우는 제외
                dup.append(i)
    
    for kw in dup:
        for idx, data in enumerate(data_list):
            indexNames = data[data['키워드']==kw].index
            data.drop(indexNames, inplace=True)
            data.to_csv(config.save_path, f'kw{idx+1}.csv')


class KoNLPy:

    def komoran_tokenizer(sent):
        komoran = Komoran()
        words = komoran.pos(sent, join=True)
        words = [w for w in words if ('/NN' in w)]
        return words

    def kkma_tokenizer(sent):
        kkma = Kkma()
        words = kkma.pos(sent, join=True)
        words = [w for w in words if ('/NN' in w)]
        return words

    def hannanum_tokenizer(sent):
        hannanum = Hannanum()
        words = hannanum.pos(sent, join=True)
        words = [w for w in words if ('N' in w)]
        return words

    def okt_tokenizer(sent):
        okt = Okt()
        words = okt.pos(sent, join=True)
        words = [w for w in words if ('Noun' in w)]
        return words
        
    def mecab_tokenizer(sent):
        mecab = Mecab()
        words = mecab.pos(sent, join=True)
        words = [w for w in words if ('NNG' in w or 'NNP' in w)]
        return words