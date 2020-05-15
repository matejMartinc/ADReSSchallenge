import pandas as pd
import librosa
import librosa.display
from collections import defaultdict
from subprocess import run
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from collections import Counter
import statistics
import argparse


import numpy as np
import os

def call_opensmile_once(wavNamePath, wavName, outputFileName, confFileName, path_output, path_opensmile):

    path_confFile = path_opensmile + "/config"

    confFile = path_confFile + '/'+ confFileName
    outputFile = path_output + '/'+ outputFileName

    # one call to opensmile
    c=[
        path_opensmile + '/bin/linux_x64_standalone_static/SMILExtract',
        '-C', confFile, '-I', wavNamePath,'-instname', wavName,
        '-csvoutput',outputFile, '-timestampcsv', str(0)
        ]
    print(' '.join(c))
    run(c)

def create_adr_features(all_features, id2idx, k):
    durations = np.array([x[0] for x in all_features])
    features = [x[1:] for x in all_features]
    clustering = KMeans(n_clusters=min(k, len(features)), random_state=0).fit(features)
    labels = clustering.labels_
    all_counts = Counter(labels)
    #print(all_counts)
    data = []
    for id, idxs in id2idx.items():
        doc_distrib = labels[idxs]
        doc_dur = durations[idxs]
        dur_dict = defaultdict(int)
        all_dur = sum(doc_dur)
        for l, dur in zip(doc_distrib, doc_dur):
            dur_dict[l] += dur
        #print(idxs, doc_distrib)

        c = Counter(doc_distrib)
        num_all = len(doc_distrib)
        counts = []
        doc_durations = []
        for i in range(k):
            if i in c:
                counts.append(c[i])
            else:
                counts.append(0)
            if i in dur_dict:
                doc_durations.append(dur_dict[i]/all_dur)
            else:
                doc_durations.append(0)
        mean = sum(counts) / len(counts)
        std = statistics.stdev(counts)

        print(id, num_all, counts)
        counts = [x/num_all for x in counts]
        features = id.split('-') + counts + doc_durations + [mean, std]
        #features = id.split('-') + doc_durations
        data.append(features)
    return data


def generate_smile_features(csvfile, audiofile):
    y, sr = librosa.load(audiofile)
    duration = librosa.get_duration(y=y, sr=sr)
    with open(csvfile, 'r', encoding='ascii') as f:
        f = f.read()
        feature_names = list(f.split('\n')[0].split(';'))[1:]
        features = list(f.split('\n')[1].split(';'))[1:]
        features = [float(x) for x in features]
        feature_names = ['duration'] + feature_names
        features = [duration] + features
    return features, feature_names


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Dementia Pit')
    argparser.add_argument('--input_path', type=str, default='data/ADReSS-IS2020-data/train/Normalised_audio-chunks', help='Path to folder with audio files')
    argparser.add_argument('--opensmile_path', type=str, default='opensmile-2.3.0', help='Path to opensmile library')
    argparser.add_argument('--output_path', type=str, default='data/adr_features_train', help='Path to folder that should contain opensmile generated features')
    argparser.add_argument('--output_tsv_path', type=str, default='tsv_features', help='Path to folder that should contain final tsv feature files used for classification')
    argparser.add_argument('--test', action='store_true', help='Generate features for test set without labels')

    args = argparser.parse_args()


    path = args.input_path
    path_output =args.output_path
    tsv_path_output = args.output_tsv_path
    path_opensmile = args.opensmile_path

    if not os.path.exists(path_output):
        os.makedirs(path_output)

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
    else:
        filelist = os.listdir(path)
        df = pd.DataFrame(filelist)
        df = df.rename(columns={0: 'file'})
    df = df.sample(frac=1, random_state=2020).reset_index(drop=True)


    confFileNames = ["gemaps/eGeMAPSv01a.conf", "MFCC12_0_D_A.conf"]
    for conf in confFileNames:
        if '/' in conf:
            fn = conf.split('/')[1].split('.')[0]
        else:
            fn = conf.split('.')[0]
        if args.test:
            fn = fn + '_test'
        else:
            fn = fn + '_train'
        counter = 0
        feature_dict = defaultdict(list)

        for idx, row in df.iterrows():
            counter += 1
            if args.test:
                file_path = os.path.join(path, row['file'])
            elif row['label'] == '0':
                file_path = os.path.join(path, 'cc', row['file'])
            else:
                file_path = os.path.join(path, 'cd', row['file'])

            outputFileName = fn + "_" + row['file'].split('.')[0] + ".tsv"

            call_opensmile_once(file_path, row['file'], outputFileName, conf, path_output, path_opensmile)
            features, feature_names = generate_smile_features(os.path.join(path_output, outputFileName), file_path)
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
            for seg in v:
                id2idx[k].append(len(all_features))
                all_features.append(np.array(seg))

        df_unfiltered = pd.DataFrame(all_features, columns=feature_names)

        print("Num features before filtering: ", df_unfiltered.shape[1])

        filtered_columns = ['duration']
        idx2col = {col:i for i, col in enumerate(feature_names)}
        for col in df_unfiltered.columns:
            if 'duration' in col:
                dur = df_unfiltered[col].tolist()
            else:
                col_data = df_unfiltered[col].tolist()
                pearson, _ = pearsonr(dur, col_data)
                if pearson < 0.2:
                    filtered_columns.append(col)

        print("Num features after filtering: ", len(filtered_columns))
        all_features_filtered = []
        keep_columns = [idx2col[col] for col in feature_names if col in filtered_columns]
        print(keep_columns)
        for f in all_features:
            f = f[keep_columns]
            all_features_filtered.append(f)

        features = create_adr_features(all_features_filtered, id2idx, num_clusters)
        if not args.test:
            feature_names = ['id', 'target'] + [fn + '_' + str(i) for i in range(len(features[0]) - 2)]
        else:
            feature_names = ['id'] + [fn + '_' + str(i) for i in range(len(features[0]) - 1)]

        print("Final num features: ", len(features[0]))

        df_data = pd.DataFrame(features, columns=feature_names)
        df_data.to_csv(os.path.join(tsv_path_output,fn + '.tsv'), encoding='utf8', sep='\t', index=False)






