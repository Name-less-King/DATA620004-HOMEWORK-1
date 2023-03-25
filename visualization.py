import numpy as np
from torch import load
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# load the model
parameters = load("./params")
w = parameters[0]
w1 = w['w1']
w2 = w['w2']

pca = PCA(n_components=10)

# visualize w1
w1_trans = pca.fit_transform(w1)
fig, axes = plt.subplots(2, 5)
scale = np.abs(w1_trans).max()
for i in range(2):
    for j in range(5):
        idx = j + i * 5
        axes[i, j].imshow(w1_trans[:, idx].reshape(28, 28),interpolation='nearest', cmap=plt.cm.Greys, vmin=-scale, vmax=scale)
        axes[i, j].axis('off')
plt.show()

# visualize w2
fig, axes = plt.subplots(2, 5)
w2_trans = w2[:-3,:].copy()
scale = np.abs(w2_trans).max()
for i in range(2):
    for j in range(5):
        idx = j + i * 5
        axes[i, j].imshow(w2_trans[:, idx].reshape(16, 17),interpolation='nearest', cmap=plt.cm.Greys, vmin=-scale, vmax=scale)
        axes[i, j].axis('off')

plt.show()

