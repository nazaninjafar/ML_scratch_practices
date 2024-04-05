import numpy as np 
from collections import Counter

class DecisionTree:
    def __init__(self, min_sample_split = 2,max_depth = 100, n_features = None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features 
        self.root = None

    def fit(self,X,y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)

    def predict(self,X):
        return np.array([self.traverse_tree(x,self.root) for x in X])

    def traverse_tree(self,x,node): 
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <=node.threshold: 
            return traverse_tree(x,node.left)
        return traverse_tree(x,node.right)

    def _grow_tree(self,X,y,depth = 0): 
        n_samples,n_features = X.shape 
        n_labels = len(np.unique(y))

        #check the stopping criteria
        if (depth>=self.max_depth or n_labels ==1 or n_samples<self.min_sample_split):
            leaf_value = self.most_common_label(y)
            return Node(value= leaf_value)
        
        feature_idx = np.random.choice(n_features,self.n_features,replace = True)
        
        #find the best split
        best_threshold, best_feature = self.best_split(X,y,feature_idx)

        #create_child nodes 
        left_idxs, right_idxs = self.split(X[:,best_feature],best_threshold)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs],depth+1)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs],depth+1)

        return Node(best_feature,best_threshold,left,right)


    def most_common_label(self,y):
        counter = Counter(y)
        print(counter.most_common())
        return counter.most_common(1)[0][0]

    def best_split(self,X,y,feat_idxs):
        best_gain = -1 
        split_idx,split_threshold = None,None 

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds: 
                #calculate information gain 
                gain = self.information_gain(y,X_column, thr)
                best_gain = gain 
                split_idx = feat_idx 
                split_threshold = thr 

        return split_threshold,split_idx

    
    def information_gain(self,y,X_column,threshold): 
        parent_entropy = self.entropy(y)

        #create children
        left_idxs,right_idxs = self.split(X_column,threshold)

        if len(left_idxs) ==0 or len(right_idxs) ==0: 
            return 0 

        #calculate the wieghted average E of children 
        left_e = self.entropy(y[left_idxs])
        right_e = self.entropy(y[right_idxs])
        n_l = len(left_idxs)
        n_r = len(right_idxs)
        n = len(y)
        child_entropy = (n_l/n)*left_e + (n_r/n) * right_e

        return parent_entropy - child_entropy




    def split(self,X_column,split_threshold): 
        left_idxs = np.argwhere(X_column<=split_threshold).flatten()
        right_idxs = np.argwhere(X_column>split_threshold).flatten()

        return left_idxs,right_idxs 



    def entropy(self,y): 
        hist = np.bincount(y)
        p_x = hist / len(y)
        return - (np.sum(p*np.log(p) for p in p_x if p>0))





         

class Node:
    def __init__(self,feature= None,threshold = None,left = None,right=None,value= None):
        self.feature = feature
        self.threshold = threshold
        self.left  = left
        self.right = right
        self.value = value  

    def is_leaf(self):
        if self.value:
            return True
        elif self.value == None:
            return False 

    
