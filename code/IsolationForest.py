import numpy as np

class Node:
    __slots__ = ['depth', 'size', 'is_leaf', 'feature', 'threshold', 'left', 'right']
    
    def __init__(self, X, depth, max_depth):
        self.depth = depth
        self.size = len(X)
        self.is_leaf = True
        
        if self.size <= 1 or depth >= max_depth:
            return
            
        self.feature = np.random.randint(0, X.shape[1])
        feature_values = X[:, self.feature]
        
        if len(np.unique(feature_values)) == 1:
            return
            
        min_val, max_val = feature_values.min(), feature_values.max()
        self.threshold = np.random.uniform(min_val, max_val)
        
        left_mask = feature_values <= self.threshold
        if not np.any(left_mask) or not np.any(~left_mask):
            return
            
        self.left = Node(X[left_mask], depth + 1, max_depth)
        self.right = Node(X[~left_mask], depth + 1, max_depth)
        self.is_leaf = False
    
    def path_length(self, X):
        results = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            node = self
            depth = 0
            
            while not node.is_leaf:
                if X[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
                depth += 1
            
            results[i] = depth + self._c(node.size)
        
        return results
    
    @staticmethod
    def _c(n):
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


class IsolationTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.feature_indices = None
    
    def fit(self, X):
        if self.max_depth is None:
            self.max_depth = int(np.ceil(np.log2(len(X))))
        
        self.root = Node(X, 0, self.max_depth)
        return self
    
    def path_length(self, X):
        return self.root.path_length(X)


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1, 
                 max_features=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.threshold = None
        
    def _get_max_samples(self, n_samples):
        if isinstance(self.max_samples, float):
            return min(int(self.max_samples * n_samples), n_samples)
        return min(self.max_samples, n_samples)
    
    def _get_max_features(self, n_features):
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return min(self.max_features, n_features)
    
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        max_samples = self._get_max_samples(n_samples)
        max_features = self._get_max_features(n_features)
        
        self.trees = []
        for i in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, max_samples, replace=False)
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            
            X_sample = X[np.ix_(sample_indices, feature_indices)]
            
            tree = IsolationTree()
            tree.fit(X_sample)
            tree.feature_indices = feature_indices
            self.trees.append(tree)

        scores = self.decision_function(X)
        self.threshold = np.quantile(scores,self.contamination)
        
        return self
    
    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        
        path_lengths = np.zeros((self.n_estimators, X.shape[0]))
        
        for i, tree in enumerate(self.trees):
            X_subset = X[:, tree.feature_indices]
            path_lengths[i] = tree.path_length(X_subset)
        
        avg_path_length = np.mean(path_lengths, axis=0)
        
        c = self._c(self.max_samples)
        return (0.5 - 2.0 ** (-avg_path_length / c))
    
    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores < self.threshold, -1, 1)
    
    @staticmethod
    def _c(n):
        if n <= 1:
            return 1.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n