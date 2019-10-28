## General imports
import numpy as np
import pandas as pd
import os,inspect

# Get this current script file's directory:
loc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Set working directory
os.chdir(loc)
# from myFunctions import gen_FTN_data
# from meSAX import *

# from dtw_featurespace import *
# from dtw import dtw
# from fastdtw import fastdtw

# to avoid tk crash
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## generate data

# set random seed
np.random.seed(1)

x_mean1 = 5
y_mean1 = 5

x_mean2 = 25
y_mean2 = 13

x_mean3 = 7
y_mean3 = 20

N1 = 50
N2 = 40
N3 = 20

coords1 = np.random.uniform(0,7,(N1,2))
coords2 = np.random.uniform(0,2,(N2,2))
# coords1 = np.random.randn(N1,2) * 2
# coords2 = np.random.randn(N2,2) * 1
coords3 = np.random.randn(N3,2) * 1
outliers = np.array([15,15,23,12]).reshape(2,2)
coords = np.empty((N1+N2+N3+outliers.shape[0],2))


coords[:N1] =  coords1 + (x_mean1,y_mean1)
coords[N1:(N1+N2)] =  coords2 + (x_mean2,y_mean2)
coords[(N1+N2):-2] =  coords3 + (x_mean3,y_mean3)
coords[-2:] = outliers


MinPts = 15

threshold = 90

## Just getting the labels:

from meLOF import meLOF

# only coordinates provided
lof = meLOF(coords,MinPts,threshold)

# only distance matrix provided
from meLOF import *
from scipy.spatial import distance
dist = distance.minkowski
D = gen_dist_mat(coords,dist)
lof = meLOF(None,MinPts,threshold,D)

# get scores and labels
scores = lof.compute_scores_()
labels = lof.compute_labels_()

# or just
scores,labels = lof.scores_labels_()

## How to use in detail:

from meLOF import *

# Generate distance matrix D
from scipy.spatial import distance
dist = distance.minkowski
D = gen_dist_mat(coords,dist)

# k-distances
k = MinPts
k_distances = k_dist(D,k=k)

# reachability distances
r_dists = r_dist(coords,D,k_distances)

# Nearest neighbors    
NN, NN_dists = nearest_neighbors(coords,k,D) 

# coordinates of the k-NN:
coords[NN]

# gives k-nearest neighbors' reachability distances
r_dists[0][NN[0]]


# average reachability distances    
r_dists_k = avg_r_dists(r_dists,NN)   

# local reachability densities
lrds = 1/r_dists_k

# LOF scores    
myLOF_scores = my_LOFs(NN,lrds)
# LOF labels
myLOF_labels = labels_(myLOF_scores, 90)



##
# plot results
my_colors = np.array(['#1f77b4','#2ca02c','#ff7f0e'])
fig, ax = plt.subplots()
plt.scatter(coords[:,0],coords[:,1],color = my_colors[labels],alpha = 0.8)
# for i in range(coords.shape[0]):
#     plt.scatter(coords[:,0],coords[:,1],color = my_colors[lof_labels[i]])

# # draw circles
# radii = myLOF_scores
# for i in range(coords.shape[0]):
#     c = plt.Circle((coords[i,0],coords[i,1]),radii[i],color = my_colors[myLOF_labels[i]] ,alpha = 0.8,
#                     fill = False)
#     ax.add_artist(c)
    
# ax.set_xlim((0,20))
# ax.set_ylim((0,20))
plt.gca().set_aspect('equal', adjustable='box') # making the x and y scale the same
plt.title('My LOF')
plt.show()



























