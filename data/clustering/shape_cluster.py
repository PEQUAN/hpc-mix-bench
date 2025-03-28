# https://classix.readthedocs.io/en/latest/clustering_tutorial.html


from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics

if __name__ == "__main__":
    random_state = 1
    moons, y1 = datasets.make_moons(n_samples=1000, noise=0.05, random_state=random_state)
    blobs, y2 = datasets.make_blobs(n_samples=1500, centers=[(-0.85,2.75), (1.75,2.25)], 
                                cluster_std=0.5, random_state=random_state)
    X = np.vstack([blobs, moons])

    y2 = y2 + len(np.unique(y1))
    labels_true = np.hstack([y1, y2])
    X_new = np.hstack((X, labels_true.reshape(-1, 1)))
    X_new = pd.DataFrame(X_new)
    pd.DataFrame(X_new).to_csv(f"shape_clusters_include_y.csv", index=True, header=True)
    from sklearn.preprocessing import StandardScaler

    # measure acc
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.25, min_samples=20).fit(X)
    labels = db.labels_

    print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
    print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
    print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels):.3f}")
    print(
        "Adjusted Mutual Information:"
        f" {metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
    )
    print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")