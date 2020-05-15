# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from collections import defaultdict
from .readability_formulas import gfi, smog, sent_len, ari, dcrf, fkgl, fre
import pandas as pd


def get_affix(text):
    return " ".join([word[-4:] if len(word) >= 4 else word for word in text.split()])


#fit and transform text features, used in scikit Feature union
class doc2vec_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        texts = data_dict[self.key]
        texts = [text.split() for text in texts]
        return texts


class text_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]

class digit_feature(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        feature = data_dict[self.key].values
        return feature.reshape(-1, 1)

class digit_features(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        columns = [col for col in data_dict.keys() if col.startswith(self.key)]
        filtered_dict = {col: data_dict[col] for col in columns}
        features = pd.DataFrame(filtered_dict).to_numpy()
        return features
        scaler = preprocessing.MinMaxScaler().fit(features)
        normalized_features = scaler.transform(features)
        return normalized_features


def countWords(wordList, text):
    cnt = 0
    length = len(text.split())
    for word in text.split():
        if word.lower() in wordList:
            cnt +=1
    if length == 0:
        return 0
    return cnt/length


def createFeatures(df_data):
    df_data['affixes'] = df_data['!PAR'].map(lambda x: get_affix(x))
    df_data['unique_words'] = df_data['!PAR'].map(lambda x: len(set(x.split()))/len(x.split()))
    df_data['text_length'] = df_data['!PAR'].map(lambda x: len(x.split()))
    df_data['gfi'] = df_data['!PAR'].map(lambda x: gfi(x))
    df_data['smog'] = df_data['!PAR'].map(lambda x: smog(x))
    df_data['sent_len'] = df_data['!PAR'].map(lambda x: sent_len(x))
    df_data['ari'] = df_data['!PAR'].map(lambda x: ari(x))
    df_data['dcrf'] = df_data['!PAR'].map(lambda x: dcrf(x))
    df_data['fkgl'] = df_data['!PAR'].map(lambda x: fkgl(x))
    df_data['fre'] = df_data['!PAR'].map(lambda x: fre(x))

    return df_data


def get_general_stats(df_data):
    # get some stats
    print('all patient: ', df_data.shape[0])
    df_ad = df_data[df_data['target'].isin([1])]
    df_contr = df_data[df_data['target'].isin([0])]
    print('num. control: ', df_contr.shape[0])
    print('num. dementia: ', df_ad.shape[0])
    print('majority class. for target: ', max(df_contr.shape[0], df_ad.shape[0]) / df_data.shape[0])


def get_attribute_stats(df_data, attribute):
    df_ad = df_data[df_data['target'].isin([1])]
    df_contr = df_data[df_data['target'].isin([0])]
    print('num. control: ', df_contr.shape[0])
    print('num. cognitive decline: ', df_ad.shape[0])


    d = defaultdict(list)

    count_ad = 0
    for idx, row in df_ad.iterrows():
        attrs = row[attribute].split()
        count_ad += len(attrs)
        for attr in attrs:
            if len(d[attr]) == 0:
                d[attr] = [0,0]
            d[attr][0] += 1

    count_contr = 0
    for idx, row in df_contr.iterrows():
        attrs = row[attribute].split()
        count_contr += len(attrs)
        for attr in attrs:
            if len(d[attr]) == 0:
                d[attr] = [0,0]
            d[attr][1] += 1

    for k in d.keys():
        d[k] = d[k][0]/count_ad - d[k][1]/count_contr

    sorted_dem = sorted(d.items(), key=lambda x: x[1], reverse=True)
    for w, s in sorted_dem:
        print(w.replace('###', ' '),s)

