import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df = pd.read_csv('./Covid_Data/Concatted_Notes_GloVe_Averages_300d.csv')
groups = df.groupby("Admission Status")

note_avgs = df['GloVe Avg']
X = []
def array_from_str(instr):
    return (np.fromstring(instr[1:-1], sep=' ')).tolist()
for i in range(len(note_avgs)):
    X.append(array_from_str(note_avgs[i]))
X = np.array(X)


pca_embedded = PCA(n_components=100).fit_transform(X)
tsne_embedded = TSNE(n_components=2).fit_transform(pca_embedded)

df['TSNE_0'] = tsne_embedded[:,0]
df['TSNE_1'] = tsne_embedded[:,1]
#df['TSNE_2'] = tsne_embedded[:,2]
#df['PCA_0'] = tsne_embedded[:,0]
#df['PCA_1'] = tsne_embedded[:,1]


groups = df.groupby('Admission Status')
for name, group in groups:
    plt.scatter(group['TSNE_0'], group['TSNE_1'], label=name)

plt.legend()
plt.title("TSNE Visualization of Record Embedding Averages")
plt.savefig('./graphics/tsne_from_pca_100_300d_s')
