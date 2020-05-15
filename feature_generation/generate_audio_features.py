import pandas as pd
import librosa
import librosa.display
from collections import defaultdict
import numpy as np
import os
import argparse


def generate_features(audiofile):
    y, sr = librosa.load(audiofile)
    duration = librosa.get_duration(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
    rms = librosa.feature.rms(y=y, hop_length=int(0.010*sr))
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=int(0.010*sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=int(0.010*sr), n_fft=int(0.025*sr))

    features = []
    feature_names = ['duration', 'chroma_stft', 'rms', 'spec_cent', 'spec_bw', 'rolloff', 'zcr']
    features.extend([duration, np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])

    for idx, e in enumerate(mfcc):
        feature_names.append('mfcc_' + str(idx))
        features.append(np.mean(e))
    return features, feature_names


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Dementia Pit')
    argparser.add_argument('--input_path', type=str, default='data/ADReSS-IS2020-data/train/Normalised_audio-chunks', help='Path to folder with audio files')
    argparser.add_argument('--output_tsv_path', type=str, default='tsv_features', help='Path to folder that should contain final tsv feature files used for classification')
    argparser.add_argument('--test', action='store_true', help='Fusion of linguistic and accustic features')

    args = argparser.parse_args()

    path = args.input_path
    tsv_path_output = args.output_tsv_path

    if not os.path.exists(tsv_path_output):
        os.makedirs(tsv_path_output)

    if not args.test:
        cc_filelist = os.listdir(os.path.join(path, 'cc'))
        df_cc = pd.DataFrame(cc_filelist)
        df_cc['label']='0'
        df_cc = df_cc.rename(columns={0:'file'})

        cd_filelist = os.listdir(os.path.join(path, 'cd'))
        df_cd = pd.DataFrame(cd_filelist)
        df_cd['label']='1'
        df_cd = df_cd.rename(columns={0:'file'})

        df = pd.concat([df_cc, df_cd], ignore_index=True)
        df = df.sample(frac=1, random_state=2020).reset_index(drop=True)
    else:
        filelist = os.listdir(path)
        df = pd.DataFrame(filelist)
        df = df.rename(columns={0: 'file'})


    counter = 0
    feature_dict = defaultdict(list)

    for idx, row in df.iterrows():
        counter += 1
        #if counter > 100:
        #    break
        if args.test:
            file_path = os.path.join(path, row['file'])
        elif row['label'] == '0':
            file_path = os.path.join(path, 'cc', row['file'])
        else:
            file_path = os.path.join(path, 'cd', row['file'])

        features, feature_names = generate_features(file_path)
        id = file_path.split('/')[-1].split('-')[0].split('.')[0]
        if not args.test:
            feature_dict[id + '-' + row['label']].append(features)
        else:
            feature_dict[id].append(features)

    data = []
    max_counter = 0
    num_clusters = 30
    id2idx = defaultdict(list)
    all_features = []
    for k,v in feature_dict.items():
        v = np.array(v)
        v = np.mean(v, axis=0)
        all_features.append(k.split('-') + v.tolist())

    if not args.test:
        df_data = pd.DataFrame(all_features, columns=['id', 'target'] + feature_names)
        df_data.to_csv(os.path.join(tsv_path_output, 'audio_features_train.tsv'), encoding='utf8', sep='\t', index=False)
    else:
        df_data = pd.DataFrame(all_features, columns=['id'] + feature_names)
        df_data.to_csv(os.path.join(tsv_path_output, 'audio_features_test.tsv'), encoding='utf8', sep='\t', index=False)
