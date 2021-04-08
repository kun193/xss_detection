#coding: utf-8

import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import csv
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
data_path = './data/feature_vectors'
label_path = './data/sha256_family.csv'

FEATURES_SET = [
    "feature",
    "permission",
    "activity",
    "service_receiver",
    "provider",
    "service",
    "intent",
    "api_call",
    "real_permission",
    "call",
    "url"
]

def load_data():
    """
    加载全部恶意样本和部分正常样本
    生成Android恶意代码检测的数据集
    """
    #列出所有样本名称
    file_names = os.listdir(data_path)
    #列出所有恶意样本的名称
    df_labels = pd.read_csv(label_path)
    malwares = df_labels['sha256'].values
    malwares = malwares.tolist()

    pos = []
    neg = []
    for fn in file_names:
        if fn in malwares:
            pos.append(fn)
        else:
            neg.append(fn)
    mals = pos[:]
    benigns = neg[:2600]
    #生成文件的详细路径
    mal_paths = [os.path.join(data_path, m) for m in mals]
    benign_paths = [os.path.join(data_path, b) for b in benigns]

    features = []
    for mp in tqdm(mal_paths):
        f = open(mp)
        for l in f.readlines():
            if l != '\n':
                l=l.strip()
                features.append(l)
        f.close()

    for bp in tqdm(benign_paths):
        f = open(bp)
        for l in f.readlines():
            if l != '\n':
                l=l.strip()
                features.append(l)
        f.close()
    features = list(set(features))

    x_m = []
    y_m = []
    x_b = []
    y_b = []
    for mp in tqdm(mal_paths):
        temp = np.zeros(len(features)+1)
        f = open(mp)
        for l in f.readlines():
            if l != '\n':
                l = l.strip()
                temp[features.index(l)] = 1
        temp[len(features)]=1
        x_m.append(list(temp))
        y_m.append(1)

    for bp in tqdm(benign_paths):
        temp = np.zeros(len(features)+1)
        f = open(bp)
        for l in f.readlines():
            if l != '\n':
                l = l.strip()
                temp[features.index(l)] = 1
        temp[len(features)]=0
        x_b.append(list(temp))
        y_b.append(0)

    # x_m = np.array(x_m)
    # y_m = np.array(y_m)
    # x_b = np.array(x_b)
    # y_b = np.array(y_b)
    # X = np.append(x_m, x_b, axis=0)
    # Y = np.append(y_m, y_b, axis=0)
    # print(X.shape,Y.shape)
    # x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)
    # return x_train, y_train, x_test, y_test
    # X=x_m+x_b
    # Y=y_m+y_b
    # malwares={"fea":X,"label":Y}
    # malpds = pd.DataFrame(malwares)
    # malpds.to_csv('malwares.csv',index=False)

    X=x_m+x_b
    data=pd.DataFrame(X)
    data.to_csv('malwares.csv',index=False)
if __name__=="__main__":
    load_data()