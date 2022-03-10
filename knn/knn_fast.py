import itertools
import numba
import threading
import queue
import numpy as np
from scipy.spatial.distance import cdist, _cdist_callable
from scipy.spatial.distance import minkowski

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

class KNNClassifierFast(object):
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

        try:
            distances = self.minkowski_distance(X.to_numpy(), self.X.to_numpy(), self.p)
        except:
            distances = self.minkowski_distance(X, self.X, self.p)

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
        try:
            distances = self.minkowski_distance_X(self.X.to_numpy(), self.p)
        except:
            distances = self.minkowski_distance_X(self.X, self.p)

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

    def minkowski_helper(self, q: queue.Queue, N, result):
        """
        Helper function to calculate minkowski distance. It picks a coordinate from the queue
        and it saves the result in the result matrix.
        """
        while True:
            task = q.get()

            i = task[0]
            prod = task[1]

            x = i // N
            y = i % N

            if (x == y):
                q.task_done()
                continue

            elif (x > y):
                q.task_done()
                continue

            m = minkowski(prod[0], prod[1], self.p)
            result[x, y] = m
            result[y, x] = m
            
            q.task_done()
        
    
    def compute_minkowski_distance(self, XA, XB, p, nr_of_threads):
        """
        
        """
        XA = np.asarray(XA)
        XB = np.asarray(XB)

        N = XB.shape[0]

        result = np.zeros([N, N])
        
        prod = list(itertools.product(XA, XA))

        q = queue.Queue()

        for i in range(len(prod)):
            q.put((i, prod[i]))

        for i in range(nr_of_threads):
            worker = threading.Thread(target=self.minkowski_helper, args=(q, N, result), daemon=True)
            worker.start()


        q.join()

        return result

    @staticmethod
    @numba.njit(parallel=True, fastmath=True)  #('(float64[:, :, :], uint64)', parallel=True, fastmath=True)
    def minkowski_distance_X(X, p):
        """
        Function that computes the minkowski distance between X and X.
        The numba decorators makes sure that this code is compiled to machine code.
        """
        N = X.shape[0]
        X = np.asarray(X)
        result = np.empty(shape=(N, N), dtype=np.float32)
        for i in numba.prange(N):
            for j in numba.prange(N):
                if (j > i):
                    continue

                elif (i == j):
                    result[i,j] = 0
                    continue

                u_v = X[i].astype(np.float32) - X[j].astype(np.float32)
                norm = np.linalg.norm(u_v, ord=p)
                result[i, j] = norm
                result[j, i] = norm


        return result

    @staticmethod
    @numba.njit(parallel=True, fastmath=True)
    def minkowski_distance(XA, XB, p):
        XA = np.asarray(XA)
        XB = np.asarray(XB)
        
        mA = XA.shape[0]
        mB = XB.shape[0]

        result = np.empty(shape=(mA, mB), dtype=np.float32)

        for i in numba.prange(mA):
            for j in numba.prange(mB):
                u_v = XA[i].astype(np.float32) - XB[j].astype(np.float32)

                norm = np.linalg.norm(u_v, ord=p)
                result[i, j] = norm

        return result
    
    @staticmethod
    #@numba.njit(parallel=True, fastmath=True)
    def compute_truncated_svd(X, nr_of_elements):
        """
        """
        X = np.asarray(X, dtype=np.float64)
        X = np.ascontiguousarray(X)
        U, s, VT = np.linalg.svd(X)
        U = np.ascontiguousarray(U)
        VT = np.ascontiguousarray(VT)

        Sigma = np.zeros((X.shape[0], X.shape[1]))
        Sigma = np.ascontiguousarray(Sigma)

        #Sigma[:X.shape[0], :X.shape[0]] = np.diag(s)
        Sigma[:np.diag(s).shape[0], :np.diag(s).shape[0]] = np.diag(s)
        
        Sigma = np.ascontiguousarray(Sigma)
        
        Sigma = Sigma[:, :nr_of_elements]
        VT = VT[:nr_of_elements, :]

        #B = U.dot(Sigma.dot(VT))
        T = np.dot(np.ascontiguousarray(U), np.ascontiguousarray(Sigma))

        #T = U.dot(Sigma)
        T = np.dot(np.ascontiguousarray(X), np.ascontiguousarray(VT.T))

        #T = X.dot(VT.T)

        #print(T)
        return T
