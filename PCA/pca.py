import numpy as np 

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.mean = None 

    def fit(self,X):
        #mean centering
        self.mean = np.mean(X,axis = 0)
        X = X-self.mean

        #covariance 
        cov = np.cov(X.T)

        #eighenvectors, eigen values 
        eigenvectors,eigen_values = np.linalg(eig(cov))

        #eigenvectors v = [:,i] column vector, transpose for easier calculations 
        eigenvectors = eigenvectors.T

        #sort eigenvectors 
        idxs = np.argsort(eigen_values[::-1])
        eigenvalues = eigen_values[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]



    def transform(self,X): 
        #project data 

        X = X-self.mean 
        return np.dot(X,self.components.T)




