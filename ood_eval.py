import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import random

# ftrain = np.load('/home/liuyuex/Documents/CrosSCLR/data/HR-ShanghaiTech/training/trajectories/00/train_position.npy').reshape(-1, 432)
# ftest = np.load('/home/liuyuex/Documents/CrosSCLR/data/HR-ShanghaiTech/testing/trajectories/00/train_position.npy').reshape(-1, 432)
# ftest = np.load('/home/liuyuex/Documents/CrosSCLR/food_not_known.npy')
# ftrain = np.load('/home/liuyuex/Documents/CrosSCLR/ftrain.npy')
# import pdb; pdb.set_trace()
# ftest = np.load('test_result.npy').reshape(-1, 256)
# labelstest = np.load('/home/liuyuex/Documents/CrosSCLR/data/HR-ShanghaiTech/processed/test_label.pkl', allow_pickle=True)[1]
# numerictest = np.load('/home/liuyuex/Documents/CrosSCLR/data/HR-ShanghaiTech/testing/trajectories/00/train_label.pkl', allow_pickle=True)[1]
# numerictest = np.load('numerictest.npy')

# fkinetics = np.load('/home/liuyuex/Documents/CrosSCLR/data/kinetics-skeleton/train_data.npy').reshape(-1, 432)
# ftrain = np.load('train_position.npy').reshape(-1, 432)
# ftest = np.load('test_position.npy').reshape(-1, 432)
# numerictest = np.load('test_label.pkl', allow_pickle=True)[1]

ftest = np.load('/home/liuyuex/Documents/CrosSCLR/food_not_known.npy')
ftrain = np.load('/home/liuyuex/Documents/CrosSCLR/ftrain.npy')
numerictest = np.load('numerictest.npy')


# normal_indices = np.where(numerictest == 0)[0]
# num_known_normal = int(len(normal_indices)*0.2)
# normal_indices = random.sample(list(normal_indices), num_known_normal)
# fnormal = ftest[normal_indices]


ood_indices = np.where(numerictest == 1)[0]
num_known_ood = int(len(ood_indices)*1)
ood_indices = random.sample(list(ood_indices), num_known_ood)
ftest = ftest[ood_indices]


inx_all = np.arange(0, len(ftrain))
train_indices = np.random.choice(inx_all, int(len(ftrain)*0.5))
ftrain = ftrain[train_indices]

# inx_all = np.arange(0, len(fkinetics))
# kinetics_indices = np.random.choice(inx_all, int(len(fkinetics)*0.1))
# fkinetics = fkinetics[kinetics_indices]

label_all = np.concatenate((np.zeros(len(ftrain)), np.ones((num_known_ood))))#, np.ones(len(fkinetics))*2))#, np.ones((num_known_normal))*2)) #

# fall = np.concatenate((ftrain, ftest))
# fall = np.concatenate((fall, fnormal))
import pdb; pdb.set_trace()
fall = np.concatenate((ftrain, ftest))
inx_all = np.arange(0, len(fall))
tsne_idx = np.random.choice(inx_all, int(len(fall)*0.1))
tsne = TSNE(n_components=2, random_state=0)
X, y = fall[tsne_idx], label_all[tsne_idx]
fall_2d = tsne.fit_transform(X, y)
plt.figure(figsize=(16, 10))
df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = fall_2d[:, 0]
df_subset['tsne-2d-two'] = fall_2d[:, 1]
df_subset['y'] = y
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.show()
