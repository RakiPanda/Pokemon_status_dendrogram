import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# クラスタ数を指定
n_clusters = 2  # ここでクラスタ数を指定します

# Pokemon.csvを読み込みます
df = pd.read_csv('Pokemon.csv')

# データの前処理（例：特定のカラムのみを使用）
# 必要なカラムのみを選択します
# ここではステータス値を含むカラムを使用する例です
features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
data = df[features]
names = df['Name']

# データのスケーリング
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# 階層的クラスタリングモデルを定義します
model = AgglomerativeClustering(metric='euclidean', 
                                linkage='ward', 
                                n_clusters=n_clusters)
model = model.fit(scaled_data)

# クラスタリングのラベルを取得
labels = model.labels_

# クラスタリング結果をデータフレームに追加
df['Cluster'] = labels

# 必要なカラムのみを選択
selected_columns = ['Name', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Cluster']
df_selected = df[selected_columns]

# クラスタ数を含むファイル名を生成
csv_filename = f'pokemon_clusters_{n_clusters}.csv'
sorted_csv_filename = f'pokemon_clusters_sorted_{n_clusters}.csv'
stats_csv_filename = f'pokemon_clusters_stats_{n_clusters}.csv'

# 各ポケモンがどのクラスタに属するかをCSVファイルに保存
df_selected.to_csv(csv_filename, index=False)

# クラスターごとにソートしたCSVファイルを保存
df_sorted = df_selected.sort_values(by='Cluster')
df_sorted.to_csv(sorted_csv_filename, index=False)

# クラスタごとの統計情報を計算して保存
stats_combined = pd.DataFrame()

for cluster_id in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    cluster_stats = cluster_data[features].describe().transpose().reset_index()
    cluster_stats['Cluster'] = cluster_id
    cluster_stats['Count'] = len(cluster_data)
    stats_combined = pd.concat([stats_combined, cluster_stats])

# クラスタごとの統計情報を保存
stats_combined.to_csv(stats_csv_filename, index=False)

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

# 階層的クラスタリングモデルを再定義（デンドログラム用）
model = AgglomerativeClustering(metric='euclidean', 
                                linkage='ward', 
                                distance_threshold=0, 
                                n_clusters=None)
model = model.fit(scaled_data)

# デンドログラムをプロットします
plt.figure(figsize=(10, 7))
plot_dendrogram(model, truncate_mode='lastp', p=n_clusters, labels=names.values)
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.show()
