import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data_dict = np.load("mpc_data.npy", allow_pickle=True).item()
for k, v in data_dict.items():
    if len(np.shape(v)) == 1:
        v = np.expand_dims(v, -1)
    elif len(np.shape(v)) == 3:
        v = v.reshape(v.shape[0], v.shape[1] * v.shape[2])
    X_tsne = TSNE(n_components=2).fit_transform(v)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])
    plt.savefig("vis/{}.png".format(k))
    print("finish vis for ", k)
