##
import numpy as np
import pandas as pd



## Common functions

def gen_FTN_data(varNames,days,filePath,fileNames):
    '''
    Input:
        - varNames: a list variable that has all the variable names
        - days: a list variable that lists out the days
    Output:
        - FTN: a dictionary with variable names as keys, and a TN format DataFrame as values
    '''
    F = len(varNames)
    N = len(days)
    # initialize output FTN dictionary with variable names as keys and empty lists as values
    FTN = dict([(i,[]) for i in varNames])
    
    for day in days:
        # read data from disk
        file = filePath + '\\' + fileNames[day]
        data = pd.read_csv(file) # TF format
        
        # get the columns we want
        df = data[varNames]
        # time increments/intervals may not be fixed, mark the indices
        indices = data.time.round(-1).drop_duplicates().index
        # T = len(indices)
        df = df.iloc[indices]
        
        # append TS to list(NT format for now)
        for i in varNames:
            FTN[i].append(df[i])
        
        print('Finished with day {}'.format(day))    
    
    print('Preparing for data output')
    for key in FTN:
        FTN[key] = np.array(FTN[key]).T # tranpose from NT to TN format
        cols = ['day{}'.format(d) for d in days] # add column names
        FTN[key] = pd.DataFrame(FTN[key],columns = cols) # make data frame
    print('Done!')
    return(FTN)


def transform_NTF2FNT(data):
    '''    
    # This function is to transform the data format from NTF to FNT
    # data is a list of matrices
    '''
    import numpy as np
    # dimensions
    N = len(data)
    T = data[0].shape[0]
    F = data[0].shape[1]
    
    # initialize
    data_transform = [] # FNT
    
    for f in range(F):
        NTmatrix = np.zeros((N,T))
        for n in range(N):
            TFmatrix = data[n] # read input NTF
            # reconstruct NT matrix
            for t in range(T):
                NTmatrix[n,t] = TFmatrix[t,f]
        # add to FNT
        data_transform.append(NTmatrix)
    
    return(data_transform)


def transform_FNT2NTF(data):
    '''
    # This function is to transform the data format from FNT to NTF
    # data is a list of matrices
    '''
    import numpy as np
    # dimensions
    F = len(data)
    N = data[0].shape[0]
    T = data[0].shape[1]
    
    # initialize
    data_transform = [] # NTF
    
    for n in range(N):
        TFmatrix = np.zeros((T,F))
        for f in range(F):
            NTmatrix = data[f] # read input FNT
            # reconstruct TF matrix
            for t in range(T):
                TFmatrix[t,f] = NTmatrix[n,t]
        # add to FNT
        data_transform.append(TFmatrix)
    
    return(data_transform)




def gen_dist_mat(X,dist_func=None,print_ = False):
    '''
    # Generate distance matrix (N*N)
    # uses Euclidean distance if not assigned
    # format should be NT
    # T: number of time steps in the time series
    # N: number of time series samples
    '''
    import numpy as np
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



h5file = r'N:\HVAC_ModelicaModel_Data\DistanceMatrix\Dw_6h60m.h5'
dataset = 'Dw'


def gen_large_dist_mat(X,filePath,dset,dist_func=None,print_ = False):
    '''
    Same as gen_dist_mat, but saves to hdf5 for large matrices(large N)
    Distance matrix is saved in a file instead returned as variable
    # Generate distance matrix (N*N)
    # uses Euclidean distance if not assigned
    # format should be NT
    # T: number of time steps in the time series
    # N: number of time series samples
    # filePath: the file path for hdf5 file
    # dset: the dataset name for the data in hdf5, e.g. f['dset']
    '''
    import numpy as np
    from scipy.spatial import distance
    import h5py
    # X = np.array(X)
    N,T = X.shape[0],X.shape[1]
    
    if dist_func is None:
        dist = distance.minkowski
    else:
        dist = dist_func
    
    # HDF5 file initialize
    f = h5py.File(filePath,'w')
    distMat = f.create_dataset(dset,shape=(N,N),dtype=np.float16,compression='gzip')
    # initialize row array
    row = np.empty((1,N),dtype=np.float32)
    
    # intialize distance matrix
    # D = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            if i==j: # identical
                row[0,j] = 0
                # distMat[i,j] = 0
                # D[i,j] = 0
            # elif i > j: # distance matrix is symmetric, no need to compute twice
                # row[0,j] = distMat[j,i]
                # distMat[i,j] = distMat[j,i]
                # D[i,j] = D[j,i]
            else:
                row[0,j] = dist(X[i,:],X[j,:])
                # distMat = dist(X[i,:],X[j,:])
                # D[i,j] = dist(X[i,:],X[j,:])
        distMat[i,:] = row
        f.flush()
        if print_: print('{}/{} finished'.format(i+1,N))
    f.close()
    return(None)



def k_dist(D,k = 4):
    '''
    inputs:
        D: distance matrix(N by N)
        k: k-th neighbor distance
    '''
    import numpy as np
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
        # p: data point. reachability distance from p to other points in coords
        coords: other data points compared to p, in NF format, np.array((N,F))
        D: precomputed distance matrix(N by N)
        # k: k-th neighbor distance
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
    
    
    
    
    
    
    
    
    
    
    







