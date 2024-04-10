import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K=2, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # liste des indices d'échantillons pour chaque cluster
        self.clusters = [[] for _ in range(self.K)]
        # les centres (vecteur de caractéristiques moyen) pour chaque cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialiser les centroïdes
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimisation
        for _ in range(self.max_iters):
            # Assigner des échantillons aux centroïdes les plus proches (créer des clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()

            # Calculer les nouveaux centroïdes à partir des clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # vérifier si les clusters ont changé
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classer les échantillons comme l'indice de leurs clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # chaque échantillon recevra l'étiquette du cluster auquel il a été assigné
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assigner les échantillons aux centroïdes les plus proches pour créer des clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance de l'échantillon actuel à chaque centroïde
        distances = [np.linalg.norm(sample - point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assigner la valeur moyenne des clusters aux centroïdes
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances entre chaque ancien et nouveau centroïde, pour tous les centroïdes
        distances = [np.linalg.norm(centroids_old[i] - centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

# Test
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)

    k.plot()