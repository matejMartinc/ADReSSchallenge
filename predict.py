from learning.text import createFeatures
import pandas as pd
import argparse
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import os




if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Dementia Pit')
    argparser.add_argument('--input_path', type=str, default='tsv_features',
                           help='Path to folder that contain final tsv feature files used for learning')
    argparser.add_argument('--model_path', type=str, help='Path to trained model used to generate predictions')
    argparser.add_argument('--result_path', type=str, default='results',
                           help='Path to folder that should contain trained models')
    args = argparser.parse_args()

    df_text = pd.read_csv(os.path.join(args.input_path, 'text_features_test.tsv'), encoding="utf8", delimiter="\t")
    df_text = df_text.sample(frac=1, random_state=123)
    df_text = createFeatures(df_text)

    print(df_text.shape)

    print('Text features: ', " ".join(list(df_text.columns)))

    df_egemaps = pd.read_csv(os.path.join(args.input_path, 'eGeMAPSv01a_test.tsv'), encoding='utf8', sep='\t')
    df_mfcc = pd.read_csv(os.path.join(args.input_path, 'MFCC12_0_D_A_test.tsv'), encoding='utf8', sep='\t')
    df_librosa = pd.read_csv(os.path.join(args.input_path, 'audio_features_test.tsv'), encoding='utf8', sep='\t')

    df_audio = df_egemaps.merge(df_mfcc, on='id')
    df_audio = df_audio.merge(df_librosa, on='id')

    df_data = df_text.merge(df_audio, on='id')
    df_data = df_data.sort_values(by=['id'])

    print("Data shape after feature creation: ", df_data.shape)
    print("Columns: ", " ".join(list(df_data.columns)))
    print('------------------------------------------')

    ids = df_data['id']
    X = df_data.drop(['id'], axis=1)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    clf = joblib.load(args.model_path)
    preds = clf.predict(X)
    s_ids = [x + ' ' for x in ids]
    s_preds = [" " + str(x) for x in preds]
    df_results = pd.DataFrame({"ID": s_ids, "Prediction": s_preds})
    df_results.to_csv('results/' + args.model_path.split('/')[1] + '_results.csv', index=False, header=True, sep=';')




