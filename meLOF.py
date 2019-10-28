# # Get this current script file's directory:
# import os,inspect
# loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# # Set working directory
# os.chdir(loc)
# from myFunctions import gen_dist_mat

import numpy as np


## class
class meLOF:
    def __init__(self,coords,MinPts = 10,threshold = 90,D=None):
        '''
        inputs:
            coords: data matrix of coordinates, in NF format (N by F)
                    Set to None if distance matrix D is provided
            MinPts: parameter of min points used for LOF, default is 10
            threshold: threshold percentile parameter, default is 90, 0-100
            D: distance matrix (N by N), if not provided, will generate one)
        ------------------------------------
        Initializes meLOF
        needs numpy and scipy
        '''
        # import numpy as np
        
        self.coords = coords
        self.MinPts = MinPts
        self.threshold = threshold
        
        if D is None:
            from scipy.spatial import distance
            dist = distance.minkowski
            
            # # Get this current script file's directory:
            # import os,inspect
            # loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            # # Set working directory
            # os.chdir(loc)
            # from myFunctions import gen_dist_mat
            
            D = self.gen_dist_mat(coords,dist)
        
        self.D = D
        
        
        
    
    def gen_dist_mat(self,X,dist_func=None,print_ = False):
        '''
        inputs:
            X: data matrix in NT(or NF) format
            dist_func: distance measure used, if not specified, will use Euclidean distance
            print_: if set to True, will print out progress during computations
        Outputs:
            D: distance matrix (N by N)
        ---------------------------------------------------------
        # Generate distance matrix (N*N)
        # uses Euclidean distance if not assigned
        # format should be NT
        # T: number of time steps in the time series
        # N: number of time series samples
        '''
        # import numpy as np
        from scipy.spatial import distance
        X = np.array(X)
        N,T = X.shape[0],X.shape[1]
        
        if dist_func is None:
            dist = distance.minkowski
        else:
            dist = dist_func
        
        # intialize distance matrix
        D = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i==j: # identical
                    D[i,j] = 0
                elif i > j: # distance matrix is symmetric, no need to compute twice
                    D[i,j] = D[j,i]
                else:
                    D[i,j] = dist(X[i,:],X[j,:])
            if print_: print('{}/{} finished'.format(i+1,N))
        return(D)
        
        
    
    def k_dist(self,D,k = 4):
        '''
        inputs:
            D: distance matrix(N by N)
            k: k-th neighbor distance, default is 4
        '''
        # import numpy as np
        D = np.array(D)
        N = D.shape[0]
        # initialize k_dist vector
        k_dist = np.zeros((N,1))
        for i in range(N):
            row = list(D[i,:])
            for j in range(k): # remove min(row) k times, not k-1 times, because closest is always itself!
                row.remove(min(row))
            k_dist[i] = min(row)
        return(k_dist)
    
    
    
    def r_dist(self,D,k_distances,dist=None):
        '''
        inputs:
            D: precomputed distance matrix(N by N)
            k_distances: a list of precomputed k-distances.
            dist: distance function
        Outputs:
            r_dists: reachability distance matrix (p,o), distance of p(rows) from o(cols)
        '''
        # import numpy as np
        if dist is None:
            from scipy.spatial import distance
            dist = distance.minkowski
        
        # k_distances = k_dist(D,k=k)
        # reachability distances
        N = D.shape[0]
        r_dists = np.empty((N,N),dtype = float)
        
        for i in range(N):
            for j in range(N):
                r_dists[i,j] = np.max([k_distances[j],D[i,j]])
        # for i,p in enumerate(coords):
        #     for j,o in enumerate(coords):
        #         r_dists[i,j] = np.max([k_distances[j],dist(p,o)])
        
        return(r_dists)
        
    # def r_dist(self,coords,D,k_distances,dist=None):
    #     '''
    #     inputs:
    #         coords: other data points compared to p, in NF format, np.array((N,F))
    #         D: precomputed distance matrix(N by N)
    #         k_distances: a list of precomputed k-distances.
    #         dist: distance function
    #     Outputs:
    #         r_dists: reachability distance matrix (p,o), distance of p(rows) from o(cols)
    #     '''
    #     # import numpy as np
    #     if dist is None:
    #         from scipy.spatial import distance
    #         dist = distance.minkowski
    #     
    #     # k_distances = k_dist(D,k=k)
    #     # reachability distances
    #     N = coords.shape[0]
    #     r_dists = np.empty((N,N),dtype = float)
    #     
    #     for i,p in enumerate(coords):
    #         for j,o in enumerate(coords):
    #             r_dists[i,j] = np.max([k_distances[j],dist(p,o)])
    #     
    #     return(r_dists)
    
    
    def nearest_neighbors(self,coords,k,D=None):
        '''
        inputs:
            coords: data coordinates in NF format, ignored if distance matrix D is provided
            k: Parameter MinPts, the k-nearest neighbors
            D: distance matrix, if not given, will use gen_dist_mat to generate one
        Outputs:
            NN_dists: k nearest neighbors distances matrix, np.array, (N by k)
            NN: k nearest neighbors matrix, np.array, (N by k)
                Contains the indices of coords, NOT the coordinates themselves
        '''
        # import numpy as np
        
        if D is None:
            from scipy.spatial import distance
            # from myFunctions import gen_dist_mat
            dist = distance.minkowski
            D = gen_dist_mat(coords,dist)
        
        N = D.shape[0]
        # initialize nearest neighbors
        NN_dists = np.empty((N,k),dtype=float)
        NN = np.empty((N,k),dtype=int)
        
        for i in range(N):
            # use numpy's structured array for sorting
            dtype = [('distance',float),('index',int)]
            structure_dist = np.empty((N,),dtype=dtype)
            structure_dist['distance'] = D[i]
            structure_dist['index'] = np.arange(N)
            structure_dist = np.sort(structure_dist,order='distance')
            
            # starts from 1 to remove itself, since the distance to itself is always 0
            NN_dists[i] = structure_dist['distance'][1:k+1] 
            NN[i] = structure_dist['index'][1:k+1] 
        
        return([NN,NN_dists])
    
    # Old version is wrong at removing step:
    
    # def nearest_neighbors(self,coords,k,D=None):
    #     '''
    #     inputs:
    #         coords: data coordinates in NF format, ignored if distance matrix D is provided
    #         k: Parameter MinPts, the k-nearest neighbors
    #         D: distance matrix, if not given, will use gen_dist_mat to generate one
    #     Outputs:
    #         NN_dists: k nearest neighbors distances matrix, np.array, (N by k)
    #         NN: k nearest neighbors matrix, np.array, (N by k)
    #             Contains the indices of coords, NOT the coordinates themselves
    #     '''
    #     # import numpy as np
    #     
    #     if D is None:
    #         from scipy.spatial import distance
    #         # from myFunctions import gen_dist_mat
    #         dist = distance.minkowski
    #         D = gen_dist_mat(coords,dist)
    #     
    #     N = D.shape[0]
    #     # initialize nearest neighbors
    #     NN_dists = np.empty((N,k),dtype=float)
    #     NN = np.empty((N,k),dtype=int)
    #     
    #     for i in range(N):
    #         dist_row = list(D[i])
    #         # remove itself, since the distance to itself is always 0
    #         dist_row.remove(min(dist_row))
    #         
    #         for j in range(k):
    #             NN_dists[i,j] = np.min(dist_row)
    #             NN[i,j] = np.argmin(dist_row)
    #             dist_row.remove(np.min(dist_row)) # remove closest point, then iterate
    #     
    #     return([NN,NN_dists])
    
    
    def avg_r_dists(self,r_dists,NN):
        '''
        inputs:
            r_dists: reachability distance matrix (p,o), distance of p(rows) from o(cols)
            NN: k-nearest neighbors matrix, np.array, (N by k)
                Contains the indices, NOT the coordinates themselves
        Outputs:
            avg_r_dists: list of average reachable distances
        '''
        # import numpy as np
        N = NN.shape[0] # number of samples
        k = NN.shape[1] # number of neighbors
        
        # initialize
        avg_r_dists = np.empty((N,),dtype=float)
        
        for i in range(N):
            avg_r_dists[i] = np.mean(r_dists[i][NN[i]])
        
        return(avg_r_dists)


    def my_LOFs(self,NN,lrds):
        '''
        inputs:
            NN: k-nearest neighbors matrix, np.array, (N by k)
                Contains the indices, NOT the coordinates themselves
            lrds: list of local reachability densities
        Outputs:
            LOFs: Local outlier factor scores
        '''
        # import numpy as np
        
        N = NN.shape[0]
        k = NN.shape[1]
        # initialize
        LOFs = np.empty((N,),dtype=float)
        
        for i in range(N):
            neighbor_lrds = lrds[NN[i]]
            numerator = np.sum(neighbor_lrds)
            lrd = lrds[i]
            LOFs[i] = numerator/lrd/k
        
        return(LOFs)


    def labels_(self,myLOF_scores,threshold_percentile = 90):
        '''
        inputs:
            myLOF_scores: LOF scores from function my_LOFs
            threshold_percentile: threshold percentile parameter, default is 90
        Outputs:
            myLOF_labels: 1 for inliers and -1 for outliers
        
        -----------------------------------------------------------------------------
        LOF in sklearn, contamination sets the threshold, default is 0.1
        check source code of fit_predict function:
        
        self.threshold_ = -scoreatpercentile(
                    -self.negative_outlier_factor_, 100. * (1. - self.contamination))
        
        '''
        # check percentiles - set up threshold
        my_lof_threshold = np.percentile(myLOF_scores,threshold_percentile)
        myLOF_scores[myLOF_scores > my_lof_threshold]
        # LOF labels
        myLOF_labels = np.empty(myLOF_scores.shape,dtype=int)
        for i,score in enumerate(myLOF_scores):
            if score > my_lof_threshold:
                myLOF_labels[i] = -1
            else:
                myLOF_labels[i] = 1
        
        return(myLOF_labels)
        
    
    def compute_scores_(self):
        '''
        outputs:
            myLOF_scores: original LOFs
        ---------------------------------------------------
        Uses default settings for everything
        
        '''
        # Generate distance matrix D
        # from scipy.spatial import distance
        # dist = distance.minkowski
        D = self.D # gen_dist_mat(coords,dist)
        coords = self.coords
        k=self.MinPts
        k_distances = self.k_dist(D,k)
        
        # reachability distances
        r_dists = self.r_dist(D,k_distances)
        
        # Nearest neighbors    
        NN, NN_dists = self.nearest_neighbors(coords,k,D) 

        # average reachability distances    
        r_dists_k = self.avg_r_dists(r_dists,NN) 
        
        # local reachability densities
        lrds = 1/r_dists_k
        
        # LOF scores    
        myLOF_scores = self.my_LOFs(NN,lrds)
        # # LOF labels
        # myLOF_labels = self.labels_(myLOF_scores, self.threshold)
        
        return(myLOF_scores)
    
    
    def compute_labels_(self):
        '''
        outputs:
            myLOF_labels: 1 for inliers and -1 for outliers
        ---------------------------------------------------
        Uses default settings for everything
        
        '''
        # # Generate distance matrix D
        # # from scipy.spatial import distance
        # # dist = distance.minkowski
        # D = self.D # gen_dist_mat(coords,dist)
        # coords = self.coords
        # k=self.MinPts
        # k_distances = self.k_dist(D,k)
        # 
        # # reachability distances
        # r_dists = self.r_dist(coords,D,k_distances)
        # 
        # # Nearest neighbors    
        # NN, NN_dists = self.nearest_neighbors(coords,k,D) 

        # # average reachability distances    
        # r_dists_k = self.avg_r_dists(r_dists,NN) 
        # 
        # # local reachability densities
        # lrds = 1/r_dists_k
        # 
        # # LOF scores    
        # myLOF_scores = self.my_LOFs(NN,lrds)
        myLOF_scores = self.compute_scores_()
        # LOF labels
        myLOF_labels = self.labels_(myLOF_scores, self.threshold)
        
        return(myLOF_labels)
    
    
    def scores_labels_(self):
        '''
        Outputs a tuple of myLOF_scores and myLOF_labels (myLOF_scores,myLOF_labels)
        outputs:
            myLOF_scores: original LOFs
            myLOF_labels: 1 for inliers and -1 for outliers
        ---------------------------------------------------
        Uses default settings for everything
        
        '''
        # # Generate distance matrix D
        # # from scipy.spatial import distance
        # # dist = distance.minkowski
        # D = self.D # gen_dist_mat(coords,dist)
        # coords = self.coords
        # k=self.MinPts
        # k_distances = self.k_dist(D,k)
        # 
        # # reachability distances
        # r_dists = self.r_dist(coords,D,k_distances)
        # 
        # # Nearest neighbors    
        # NN, NN_dists = self.nearest_neighbors(coords,k,D) 

        # # average reachability distances    
        # r_dists_k = self.avg_r_dists(r_dists,NN) 
        # 
        # # local reachability densities
        # lrds = 1/r_dists_k
        # 
        # # LOF scores    
        # myLOF_scores = self.my_LOFs(NN,lrds)
        myLOF_scores = self.compute_scores_()
        # LOF labels
        myLOF_labels = self.labels_(myLOF_scores, self.threshold)
        
        return(myLOF_scores,myLOF_labels)



## Pure functions
def gen_dist_mat(X,dist_func=None,print_ = False):
    '''
    inputs:
        X: data matrix in NT(or NF) format
        dist_func: distance measure used, if not specified, will use Euclidean distance
        print_: if set to True, will print out progress during computations
    Outputs:
        D: distance matrix (N by N)
    ---------------------------------------------------------
    # Generate distance matrix (N*N)
    # uses Euclidean distance if not assigned
    # format should be NT
    # T: number of time steps in the time series
    # N: number of time series samples
    '''
    # import numpy as np
    from scipy.spatial import distance
    X = np.array(X)
    N,T = X.shape[0],X.shape[1]
    
    if dist_func is None:
        dist = distance.minkowski
    else:
        dist = dist_func
    
    # intialize distance matrix
    D = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i==j: # identical
                D[i,j] = 0
            elif i > j: # distance matrix is symmetric, no need to compute twice
                D[i,j] = D[j,i]
            else:
                D[i,j] = dist(X[i,:],X[j,:])
        if print_: print('{}/{} finished'.format(i+1,N))
    return(D)
    
    

def k_dist(D,k = 4):
    '''
    inputs:
        D: distance matrix(N by N)
        k: k-th neighbor distance, default is 4
    '''
    # import numpy as np
    D = np.array(D)
    N = D.shape[0]
    # initialize k_dist vector
    k_dist = np.zeros((N,1))
    for i in range(N):
        row = list(D[i,:])
        for j in range(k): # remove min(row) k times, not k-1 times, because closest is always itself!
            row.remove(min(row))
        k_dist[i] = min(row)
    return(k_dist)


def r_dist(coords,D,k_distances,dist=None):
    '''
    inputs:
        coords: other data points compared to p, in NF format, np.array((N,F))
        D: precomputed distance matrix(N by N)
        k_distances: a list of precomputed k-distances.
        dist: distance function
    Outputs:
        r_dists: reachability distance matrix (p,o), distance of p(rows) from o(cols)
    '''
    import numpy as np
    if dist is None:
        from scipy.spatial import distance
        dist = distance.minkowski
    
    # k_distances = k_dist(D,k=k)
    # reachability distances
    N = coords.shape[0]
    r_dists = np.empty((N,N),dtype = float)
    
    for i,p in enumerate(coords):
        for j,o in enumerate(coords):
            r_dists[i,j] = np.max([k_distances[j],dist(p,o)])
    
    return(r_dists)



def nearest_neighbors(coords,k,D=None):
    '''
    inputs:
        coords: data coordinates in NF format
        k: Parameter MinPts, the k-nearest neighbors
        D: distance matrix, if not given, will use gen_dist_mat to generate one
    Outputs:
        NN_dists: k nearest neighbors distances matrix, np.array, (N by k)
        NN: k nearest neighbors matrix, np.array, (N by k)
            Contains the indices of coords, NOT the coordinates themselves
    '''
    import numpy as np
    
    if D is None:
        from scipy.spatial import distance
        from myFunctions import gen_dist_mat
        dist = distance.minkowski
        D = gen_dist_mat(coords,dist)
    
    N = D.shape[0]
    # initialize nearest neighbors
    NN_dists = np.empty((N,k),dtype=float)
    NN = np.empty((N,k),dtype=int)
    
    for i in range(N):
        # use numpy's structured array for sorting
        dtype = [('distance',float),('index',int)]
        structure_dist = np.empty((N,),dtype=dtype)
        structure_dist['distance'] = D[i]
        structure_dist['index'] = np.arange(N)
        structure_dist = np.sort(structure_dist,order='distance')
        
        # starts from 1 to remove itself, since the distance to itself is always 0
        NN_dists[i] = structure_dist['distance'][1:k+1] 
        NN[i] = structure_dist['index'][1:k+1] 
    
    return([NN,NN_dists])

# def nearest_neighbors(coords,k,D=None):
#     '''
#     inputs:
#         coords: data coordinates in NF format
#         k: Parameter MinPts, the k-nearest neighbors
#         D: distance matrix, if not given, will use gen_dist_mat to generate one
#     Outputs:
#         NN_dists: k nearest neighbors distances matrix, np.array, (N by k)
#         NN: k nearest neighbors matrix, np.array, (N by k)
#             Contains the indices of coords, NOT the coordinates themselves
#     '''
#     import numpy as np
#     
#     if D is None:
#         from scipy.spatial import distance
#         from myFunctions import gen_dist_mat
#         dist = distance.minkowski
#         D = gen_dist_mat(coords,dist)
#     
#     N = D.shape[0]
#     # initialize nearest neighbors
#     NN_dists = np.empty((N,k),dtype=float)
#     NN = np.empty((N,k),dtype=int)
#     
#     for i in range(N):
#         dist_row = list(D[i])
#         # remove itself, since the distance to itself is always 0
#         dist_row.remove(min(dist_row))
#         
#         for j in range(k):
#             NN_dists[i,j] = np.min(dist_row)
#             NN[i,j] = np.argmin(dist_row)
#             dist_row.remove(np.min(dist_row)) # remove closest point, then iterate
#     
#     return([NN,NN_dists])
    
    


def avg_r_dists(r_dists,NN):
    '''
    inputs:
        r_dists: reachability distance matrix (p,o), distance of p(rows) from o(cols)
        NN: k-nearest neighbors matrix, np.array, (N by k)
            Contains the indices, NOT the coordinates themselves
    Outputs:
        avg_r_dists: list of average reachable distances
    
    '''
    import numpy as np
    N = NN.shape[0] # number of samples
    k = NN.shape[1] # number of neighbors
    
    # initialize
    avg_r_dists = np.empty((N,),dtype=float)
    
    for i in range(N):
        avg_r_dists[i] = np.mean(r_dists[i][NN[i]])
    
    return(avg_r_dists)



def my_LOFs(NN,lrds):
    '''
    inputs:
        NN: k-nearest neighbors matrix, np.array, (N by k)
            Contains the indices, NOT the coordinates themselves
        lrds: list of local reachability densities
    Outputs:
        LOFs: Local outlier factor scores
    '''
    import numpy as np
    
    N = NN.shape[0]
    k = NN.shape[1]
    # initialize
    LOFs = np.empty((N,),dtype=float)
    
    for i in range(N):
        neighbor_lrds = lrds[NN[i]]
        numerator = np.sum(neighbor_lrds)
        lrd = lrds[i]
        LOFs[i] = numerator/lrd/k
    
    return(LOFs)
    
    
def labels_(myLOF_scores,threshold_percentile = 90):
    '''
    inputs:
        myLOF_scores: LOF scores from function my_LOFs
        threshold_percentile: threshold percentile parameter, default is 90
    Outputs:
        myLOF_labels: 1 for inliers and -1 for outliers
    
    -----------------------------------------------------------------------------
    LOF in sklearn, contamination sets the threshold, default is 0.1
    check source code of fit_predict function:
    
    self.threshold_ = -scoreatpercentile(
                -self.negative_outlier_factor_, 100. * (1. - self.contamination))
    
    '''
    # check percentiles - set up threshold
    my_lof_threshold = np.percentile(myLOF_scores,threshold_percentile)
    myLOF_scores[myLOF_scores > my_lof_threshold]
    # LOF labels
    myLOF_labels = np.empty(myLOF_scores.shape,dtype=int)
    for i,score in enumerate(myLOF_scores):
        if score > my_lof_threshold:
            myLOF_labels[i] = -1
        else:
            myLOF_labels[i] = 1
    
    return(myLOF_labels)

## How to use

# # Generate distance matrix D
# from scipy.spatial import distance
# dist = distance.minkowski
# D = gen_dist_mat(coords,dist)
# 
# # reachability distances
# r_dists = r_dist(coords,D,k_distances)
# 
# # Nearest neighbors    
# NN, NN_dists = nearest_neighbors(coords,k,D) 
# 
# # coordinates of the k-NN:
# coords[NN]
# 
# # gives k-nearest neighbors' reachability distances
# r_dists[0][NN[0]]
# 
# 
# # average reachability distances    
# r_dists_k = avg_r_dists(r_dists,NN)   
# 
# # local reachability densities
# lrds = 1/r_dists_k
# 
# # LOF scores    
# myLOF_scores = my_LOFs(NN,lrds)
# # LOF labels
# myLOF_labels = labels_(myLOF_scores, 90)















