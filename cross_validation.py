from learning.text import createFeatures, text_col, get_general_stats, digit_feature, digit_features, doc2vec_transformer
import pandas as pd
import pickle
import argparse
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import model_selection
from sklearn import pipeline
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.metrics import make_scorer, mean_squared_error
import math
import numpy as np
from gensim.sklearn_api import D2VTransformer
import os



def get_mmse_scores(input_path1, input_path2=None):
    df1 = pd.read_csv(input_path1, sep=';')
    if input_path2 is not None:
        df2 = pd.read_csv(input_path2, sep=';')
        df = pd.concat([df1, df2])
    else:
        df = df1
    df = df.drop(['age', 'gender'], axis=1)
    df['id'] = df['id'].map(lambda x: x.strip())
    return df



def powerset(s):
    all = []
    x = len(s)
    for i in range(1, 1 << x):
        all.append([s[j] for j in range(x) if (i & (1 << j))])
    return all


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Dementia Pit')
    argparser.add_argument('--input_path', type=str, default='tsv_features', help='Path to folder that contain final tsv feature files used for classification')
    argparser.add_argument('--output_model_path', type=str, default='trained_models', help='Path to folder that should contain trained models')
    argparser.add_argument('--meta_path', type=str, default='data/ADReSS-IS2020-data/train/cc_meta_data.txt;data/ADReSS-IS2020-data/train/cd_meta_data.txt',
                           help='Path to both meta files containing MMSE score for non-AD and patients separated with ";"')
    argparser.add_argument('--regression', action='store_true', help='Fusion of linguistic and accustic features')
    argparser.add_argument('--feature_set', type=str, default='each', choices=['each', 'all_combs', 'one_comb'],
                           help='each: return results for each feature, all_combs: return results for all combination of features, one_comb: return results for one combination of features')
    args = argparser.parse_args()

    if args.regression:
        task = 'regression'
    else:
        task = 'classification'

    df_text = pd.read_csv(os.path.join(args.input_path, 'text_features_train.tsv'), encoding="utf8", delimiter="\t")
    df_text = createFeatures(df_text)
    df_text['target'] = df_text['target'].map(lambda x: 0 if x == '!CC' else 1)

    if args.regression:
        meta_1, meta_2 = args.meta_path.split(';')
        df_mmse = get_mmse_scores(meta_1, meta_2)
        df_text = df_text.merge(df_mmse, on='id')

    print(df_text.shape)

    print('Text features: ', " ".join(list(df_text.columns)))


    df_egemaps = pd.read_csv(os.path.join(args.input_path, 'eGeMAPSv01a_train.tsv'), encoding='utf8', sep='\t')
    df_egemaps = df_egemaps.drop(['target'], axis=1)
    df_mfcc = pd.read_csv(os.path.join(args.input_path, 'MFCC12_0_D_A_train.tsv'), encoding='utf8', sep='\t')
    df_mfcc = df_mfcc.drop(['target'], axis=1)
    df_librosa = pd.read_csv(os.path.join(args.input_path, 'audio_features_train.tsv'), encoding='utf8', sep='\t')
    df_librosa = df_librosa.drop(['target'], axis=1)

    df_audio = df_egemaps.merge(df_mfcc, on='id')
    df_audio = df_audio.merge(df_librosa, on='id')
    df_data = df_text.merge(df_audio, on='id')
    df_data = df_data.sample(frac=1, random_state=123)

    if not os.path.exists(args.output_model_path):
        os.makedirs(args.output_model_path)



    print('------------------------------------------')
    print('Stats:')
    get_general_stats(df_data)
    print('------------------------------------------')

    print("Data shape after feature creation: ", df_data.shape)
    print("Columns: ", " ".join(list(df_data.columns)))
    print('------------------------------------------')

    if args.regression:
        y = df_data['mmse'].values
    else:
        y = df_data['target'].values
    X = df_data.drop(['id', 'target'], axis=1)


    # build classification model
    if args.regression:
        lr_1 = LinearRegression(fit_intercept=True)
        svm_10 = LinearSVR(C=10, fit_intercept=True, random_state=123, max_iter=10000)
        svm_100 = LinearSVR(C=100, fit_intercept=True, random_state=123, max_iter=10000)
        xgb = xgb.XGBRegressor(max_depth=10, subsample=0.8, n_estimators=50, colsample_bytree=0.8, learning_rate=1, nthread=8)
        rfc = RandomForestRegressor(random_state=123, n_estimators=50, max_depth=5)
        learners = [('xgb',xgb), ('rfc',rfc), ('svm_10', svm_10), ('svm_100', svm_100), ('lr_1',lr_1)]
    else:
        svm_10 = LinearSVC(penalty='l2', multi_class='ovr', fit_intercept=True, C=10, max_iter=10000)
        lr_10 = LogisticRegression(C=10, solver='liblinear', fit_intercept=True, random_state=123)
        lr_100 = LogisticRegression(C=100, solver='liblinear', fit_intercept=True, random_state=123)
        rfc = RandomForestClassifier(random_state=123, n_estimators=50, max_depth=5)
        xgb = xgb.XGBClassifier(max_depth=10, subsample=0.8, n_estimators=50, colsample_bytree=0.8, learning_rate=1, nthread=8)
        learners = [('svm_10',svm_10)]

    tfidf_unigram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=1, max_df=0.8)
    tfidf_bigram = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=False, min_df=1, max_df=0.5)
    tfidf_pos = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=True, min_df=1, max_df=0.6, lowercase=False)
    tfidf_ud = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1, max_df=0.6, lowercase=False)
    tfidf_gra = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1, max_df=0.6, lowercase=False)
    character_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), lowercase=False, min_df=1, max_df=0.8)
    bigram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2), lowercase=False, min_df=4, max_df=0.8)
    tfidf_ngram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=0.1, max_df=0.6)
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    tsvd = TruncatedSVD(random_state=2016, n_components=200, n_iter=5)
    doc2vec = D2VTransformer()

    all_features = [('ARI', digit_feature(key='ari')),
        ('GFI', digit_feature(key='gfi')),
        ('smog', digit_feature(key='smog')),
        ('Num. unique words', digit_feature(key='unique_words')),


        ('eGeMAPS ADR', digit_features(key='eGeMAPS')),
        ('MFCC ADR', digit_features(key='MFCC12')),
        ('MFCC', digit_features(key='mfcc')),
        ('duration', digit_features(key='duration')),

        ('Unigram', pipeline.Pipeline([('s1', text_col(key='!PAR')), ('tfidf_unigram', tfidf_unigram)])),
        ('Bigram', pipeline.Pipeline([('s2', text_col(key='!PAR')), ('tfidf_bigram', tfidf_bigram)])),
        ('Character 4-grams', pipeline.Pipeline([('s5', text_col(key='!PAR')), ('character_vectorizer', character_vectorizer),
                            ('tfidf_character', tfidf_transformer)])),
        ('Suffixes', pipeline.Pipeline([('s6', text_col(key='affixes')), ('tfidf_ngram', tfidf_ngram)])),
        ('POS tag', pipeline.Pipeline([('s4', text_col(key='!MORPH_PAR')), ('tfidf_pos', tfidf_pos)])),

        ('GRA',pipeline.Pipeline([('s9', text_col(key='!GRA_PAR')), ('tfidf_gra', tfidf_ud)])),
        ('UD', pipeline.Pipeline([('s10', text_col(key='!UD_PAR')), ('tfidf_ud', tfidf_ud)])),
        ('doc2vec UD', pipeline.Pipeline([('s11', doc2vec_transformer(key='!UD_PAR')), ('doc2vec', doc2vec)])),
    ]

    if args.feature_set == 'all_combs':
        all_features = powerset(all_features)

    best_feat_combo = []
    best_learner = ''
    counter = 0
    len_all_combs = len(all_features)

    if args.feature_set == 'all_combs':
        print('Num all feature combinations: ', len_all_combs)
    print()

    for name, learner in learners:
        names = []
        scores = []
        counter = 0
        best_acc = 0
        best_rmse = 9999999999999999999999

        print('Testing learner: ', name)

        for features in all_features:
            if args.feature_set == 'each':
                fname = features[0]
                names.append(fname)
                features = [features]
            elif args.feature_set == 'one_comb':
                features = all_features

            counter += 1
            clf = pipeline.Pipeline([
                ('union', FeatureUnion(
                    transformer_list=features,
                    n_jobs=1
                )),
                #('scale', Normalizer()),
                ('lr', learner)])

            kfold = model_selection.KFold(n_splits=10)
            if args.regression:
                rmse_scorer = make_scorer(mean_squared_error)
                results = model_selection.cross_val_score(clf, X, y, cv=kfold, verbose=0, scoring=rmse_scorer)
            else:
                results = model_selection.cross_val_score(clf, X, y, cv=kfold, verbose=0, scoring='accuracy')
            feat_comb = [x[0] for x in features]

            clf.fit(X, y)

            if args.regression:
                rmse = np.array([math.sqrt(x) for x in results])
                rmse = rmse.mean()
                scores.append(rmse)
                if rmse <= best_rmse:
                    best_learner = name
                    best_rmse = rmse
                    best_feat_combo = feat_comb
                    pickle.dump(clf, open(
                        os.path.join(args.output_model_path, 'model_alg:' + name + '_rmse:' + str(rmse)[:5] + '_feat:' + "_".join(
                            feat_comb) + '_' + task + '.pkl'), 'wb'))
                print()
                print('Feat combo', str(counter) + '/' + str(len_all_combs) + ':', feat_comb)
                print('Learner: ', name)
                print('RMSE: ', rmse)
                print('Best RMSE: ', best_rmse)
                print('Best feat combo: ', best_feat_combo)
                print('Best learner: ', best_learner)
            else:
                acc = results.mean()
                scores.append(acc)
                if acc >= best_acc:
                    best_learner = name
                    best_acc = acc
                    best_feat_combo = feat_comb
                    pickle.dump(clf, open(os.path.join(args.output_model_path, 'model_alg:' + name + '_acc:' + str(acc)[:5] + '_feat:' + "_".join(feat_comb) + '_' + task + '.pkl'), 'wb'))
                print()
                print('Feat combo', str(counter) + '/' + str(len_all_combs) + ':', feat_comb)
                print('Learner: ', name)
                print('Accuracy: ', acc)
                print('Best accuracy: ', best_acc)
                print('Best feat combo: ', best_feat_combo)
                print('Best learner: ', best_learner)
            print('---------------------------------------------------------------------')
            print()
            if args.feature_set == 'one_comb':
                break
        if args.feature_set=='each':
            print(names)
            print(scores)


#['ARI', 'GFI', 'smog', 'Num. unique words', 'eGeMAPS ADR', 'MFCC ADR', 'MFCC', 'duration', 'Unigram', 'Bigram', 'Character 4-grams', 'Suffixes', 'POS tag', 'GRA', 'UD', 'doc2vec UD']
#[0.6209090909090909, 0.509090909090909, 0.5872727272727272, 0.5481818181818181, 0.5472727272727271, 0.5545454545454546, 0.5063636363636362, 0.5554545454545454, 0.7990909090909091, 0.7054545454545456, 0.8636363636363636, 0.8072727272727273, 0.7054545454545456, 0.7072727272727273, 0.6954545454545454, 0.5245454545454545]



