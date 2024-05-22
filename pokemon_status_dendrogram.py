import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Pokemon.csvを読み込みます
df = pd.read_csv('Pokemon.csv')

# データの前処理（例：特定のカラムのみを使用）
# 必要なカラムのみを選択します
# ここではステータス値を含むカラムを使用する例です
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
data = df[features]

# データのスケーリング
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 階層的クラスタリングモデルを定義します
model = AgglomerativeClustering(metric='euclidean', 
                                linkage='ward', 
                                distance_threshold=0, 
                                n_clusters=None)
model = model.fit(scaled_data)

# デンドログラムをプロットする関数
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if (child_idx < n_samples):
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # デンドログラムをプロットします
    dendrogram(linkage_matrix, **kwargs)

# デンドログラムをプロットします
plt.figure(figsize=(10, 7))
plot_dendrogram(model, truncate_mode='lastp', p=6)
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.show()
