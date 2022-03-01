import numpy as np
from scipy.spatial.distance import cdist

def accuracy(Y_true: np.array, Y_predicted: np.array):
    assert Y_true.shape[0] == Y_predicted.shape[0]
    return np.mean(Y_predicted == Y_true)

def majority(x, k):
    """
    Auxiliary function to get the majority among k-nearest neighbors. It is extended to add custom tie-breaking rule
    (instead of standard picking first of satisfying numbers): weighted k-nearest neighbors (the class with smallest
    sum of distances to target)

     :param x: row, where first k cells are distances to k-nearest neighbors and last k cells are labels of k-nearest
     neigbors

     :return: value of label
    """
    distances = x[:k]
    labels = x[k:]
    values, counts = np.unique(labels, return_counts = True)
    if k == 1:
        return labels[0]
    elif (k>1) and (list(counts).count(max(list(counts))) > 1):
        labels_to_compare = values[np.where(counts == max(list(counts)))]
        weighted_dist = []
        for i in range(labels_to_compare.shape[0]):
            weighted_dist.append(np.sum(distances[np.where(labels == labels_to_compare[i])]))
        return labels_to_compare[np.argmin(weighted_dist)]
    else:
        return max(set(list(labels)), key = list(labels).count)

class KNNClassifier(object):
    """
    A class for KNN classifier
    """

    def __init__(self, k: int = 5, p: float = 2.):
        """
        """

        self.k = k
        self.p = p
        self.Y = None
        self.X = None
        self.predicted_labels = None
        self.loocv_labels = None
        self.loocv_accuracy = None

    def fit(self, X: np.ndarray, Y: np.array):
        """
        """
        # get initial data that is base for our further decision
        # training data feature space
        self.X = X
        #  training data label space
        self.Y = Y
        
        return self

    def predict(self, X: np.ndarray):
        """
        
        """
        # get distances between input X and train X
        distances = cdist(XA=X, XB=self.X, metric='minkowski', p=self.p)
        # get auxiliary label matrix
        labels = np.tile(self.Y, (X.shape[0], 1))
        supermatrix = np.zeros((X.shape[0], self.k, 2))

        # sort distances
        sorted_points_indices = np.apply_along_axis(np.argsort, 1, distances)[:, :self.k]
        supermatrix[:, :, 0] = distances[np.arange(sorted_points_indices.shape[0])[:, None], sorted_points_indices]
        # sort labels according to indices
        supermatrix[:, :, 1] = labels[np.arange(sorted_points_indices.shape[0])[:, None], sorted_points_indices]

        # predict labels using rule with tie-breaking extension
        self.predicted_labels = np.apply_along_axis(majority, 1,
                                                    supermatrix.reshape((supermatrix.shape[0],
                                                                         2 * supermatrix.shape[1]),
                                                                        order='F'), k=self.k)
        return self.predicted_labels

    
    def calculate_loocv(self):
        # get distances between input X and train X
        distances = cdist(XA=self.X, XB=self.X, metric='minkowski', p=self.p)
        # get auxiliary label matrix
        labels = np.tile(self.Y, (self.X.shape[0], 1))
        supermatrix = np.zeros((self.X.shape[0], self.k, 2))

        # sort distances
        sorted_points_indices = np.apply_along_axis(np.argsort, 1, distances)[:, 1:self.k+1]
        supermatrix[:, :, 0] = distances[np.arange(sorted_points_indices.shape[0])[:, None], sorted_points_indices]
        # sort labels according to indices
        supermatrix[:, :, 1] = labels[np.arange(sorted_points_indices.shape[0])[:, None], sorted_points_indices]

        # predict labels using rule with tie-breaking extension
        self.loocv_labels = np.apply_along_axis(majority, 1,
                                                    supermatrix.reshape((supermatrix.shape[0],
                                                                         2 * supermatrix.shape[1]),
                                                                        order='F'), k=self.k)
        self.loocv_accuracy = accuracy(self.Y, self.loocv_labels)
        return self.loocv_accuracy

    def accuracy(self, Y: np.array):
        assert self.predicted_labels.shape[0] == Y.shape[0]
        return np.sum(Y == self.predicted_labels)/Y.shape[0]