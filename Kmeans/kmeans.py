import numpy as np 

def euclidean_distance(x1,x2): 
    return np.sqrt(np.sum((x1-x2)**2))

class Kmeans: 

    def __init__(self,K=5,max_iters=100,plot_steps=False):
        self.K = K 
        self.max_iters = 100 
        self.plot_steps = plot_steps 

        #list of sample indices for each clusters
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each cluster 
        self.centroids = []

    def predict(self,X):
        self.X = X 
        self.n_samples, self.n_features = X.shape 

        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples,self.K,replace = False)
        self.centroids = [self.X[random_sample_idxs] for idx in random_sample_idxs]
        
        #Optimize clusters 
        for _ in range(self.max_iters): 
            #assign samples to the closes centroids 
            self.clusters = self._create_clusters(self.centroids)

            

            #calculate new centroids from clusters 
            centroid_old = self.centroids 
            self.centroids = self.get_centroids(self.clusters)

        if self._is_converged(centroid_old,self.centroids):
            break 
        
        if self.plot_steps:
            self.plot()

        #classify samples as the index of their clusters 
        return self._get_cluster_labels(self.clusters)


    def _create_clusters(self,centroids):
        #create the samples to the closest centroids 
        clusters = [[] for _ in range(self.K)]
        for idx,sample in enumerate(self.X): 
            centroid_idx = self._closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)

        return clusters 

    def _closest_centroid(self,sample,centroids): 
        #distance of the current sample to each centroid 
        distance = [euclidean_distance(sample.point) for point in centroids]
        closest_idx = np.argmin(distance)
        return closest_idx 




    def _get_cluster_labels(self,clusters): 
        #each sample will get the label of the cluster it is assigned 
        labels = np.empty(self.n_samples)
        for cluster_idx,cluster in enumerate(clusters): 
            for sample_idx in cluster: 
                labels[sample_idx] = cluster_idx

        return labels 

    def _get_clusters(self,clusters):
        #assign mean values to centroids 
        centroids = np.zeros((self.K,self.n_features))
        for cluster_idx,cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster],axis = 0)
            centroids[cluster_idx] = cluster_mean 
        return centroids 

    def _is_converged(self,centroid_old,centroids): 
        #distances between old and new centroids 
        distance = [euclidean_distance(centroid_old[i],centroids[i]) for i in range(self.K)]
        return sum(distance) ==0 

    



    

    
