#!/usr/bin/env python
# coding: utf-8

# Implementing GMM for MNIST dataset.  
# 
# Below is the code to Implement the EM algorithm for fitting a Gaussian mixture model for the MNIST handwritten
# digits dataset. Here, we reduce the dataset to be only two cases, of digits “2” and “6” only. 
# 
# Thus, we will fit GMM with C = 2. 
# 
# Data: data.mat & label.mat
# The matrix images is of size 784-by-1990, i.e., there are 1990 images in total, and each
# column of the matrix corresponds to one image of size 28-by-28 pixels (the image is vectorized;
# the original image can be recovered by mapping the vector into a matrix).
# 
# First, we use PCA to reduce the dimensionality of the data before applying to EM. 
# PCA of MNIST handwritten digits dataset
# 
# We will put all “6” and “2” digits together, to project the original data into 4-dimensional vectors.
# 
# Next, we implement EM algorithm for the projected data (with 4-dimensions).
# (Here we use the same set of data for training and testing)
#     
# (a) To Implement EM algorithm "by hand", we use the following initialization:
# • initialization for mean: random Gaussian vector with zero mean
# • initialization for covariance: generate two Gaussian random matrix of size n-byn:
# S1 and S2, and initialize the covariance matrix for the two components are 
# 
# $$Σ_{1}=S_{1}S_{1}^{T}+I_{n} and Σ_{2}=S_{2}S_{2}^{T}+I_{n}$$
# 
# where $I_{n}$ is an identity matrix of size n-by-n. We will plot the log-likelihood function vs. the number of iterations to show the algorithm is converging.
# 
# (b) We report the fitted GMM model when EM has terminated in the algorithms
# as follows: We report the weights for each component, and the mean of each component,
# by mapping them back to the original space and reformat the vector to make them
# into 28-by-28 matrices and show images. Ideally, we should be able to see these
# means corresponds to some kind of “average” images. We will report the two 4-by-4
# covariance matrices by visualizing their intensities (e.g., using a gray scaled image or
# heat map).
# 
# (c) We will Use the τik
# to infer the labels of the images, and compare with the true labels.
# We report the mis-classification rate for digits “2” and “6” respectively; & perform K-means
# clustering with K = 2.
# We will show the mis-classification rate for digits “2” and “6” respectively,
# and compare with GMM, & identify which one achieves the better performance.


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# In[2]:


data_raw = sio.loadmat('data.mat')  
label_raw = sio.loadmat('label.mat')  


# In[3]:


def standardize_pixel_2 (xtrain):
    xtrain_raw = data_raw[xtrain]
    xtrain_raw = xtrain_raw.astype(float) 
    xtrain_standardized = xtrain_raw/255
    return xtrain_standardized, xtrain_raw
ndata_a, data_raw2 = standardize_pixel_2('data')
ndata_a.shape, data_raw2.shape


# In[4]:


def standardize_label (xtrain):
    xtrain_raw = label_raw[xtrain]
    xtrain_raw = xtrain_raw.astype(float) 
    xtrain_standardized = xtrain_raw/255
    return xtrain_standardized, xtrain_raw
label_st, label_a = standardize_label('trueLabel')
label_st.shape, label_a.shape


# In[5]:


ndata =ndata_a.T
label_aa=label_a.T
ndata.shape, label_aa.shape


# In[6]:


label = label_aa[:,0]
label.shape


# ### PCA of MNIST handwritten digits dataset
# 
# We will put all “6” and “2” digits together, to project the original data into 4-dimensional vectors.
# Now implement EM algorithm for the projected data (with 4-dimensions).
# Use 4 PC's in anaysis:
# First use PCA to reduce the dimensionality of the data before applying to EM. 
# 
# We will put all “6” and “2” digits together, to project the original data into 4-dimensional vectors.
# 

# In[7]:


m, n = ndata.shape
C = np.matmul(ndata.T, ndata)/m
# pca the data
d = 4  # reduced dimension
values, V = np.linalg.eig(C)
#eienvalues, eienvectors, above 
ind = np.argsort(values)[::-1][:d]
V = V[:, ind]
values = values.real
V = V.real
data_r = np.matmul(ndata,V)
data_r.shape , ndata.shape,  V.shape #(784, 4)


# In[8]:


def reconstruct_data(pdata, evectors, K):
    X_reconstructed =  np.matmul(pdata[:, :K], evectors[:, :K].T)# + X_mean
    return X_reconstructed


# In[9]:


data_s = reconstruct_data(data_r, V, 4)
data_s.shape


# EDX notes: Each individual image is represented by 784x1, which is then being reduced to 4x1. Since you have 1990 images, the ending shape should be 4x1990.
# 
# Dimensions of the the eigen vector matrix are original dimensions x reduced dimensions which in this case is 784 x 4. When you multiply the original data matrix (1900x784) with eigen vector matrix (784x4), dimensions of the product become 1990x4.
# 
# We will report the following for b. weights for each component: 
# mu and reshaped back to show the images; sigma in 4 x 4  as a heat map or grey scale image 
# 
# 
# The "weights" are designated by PI(k), where k is the cluster number.  In this case PI has 2 values (one for the "2" and one for the "6").
# 

# In[10]:


data = data_s.T
data.shape


# Now implement EM algorithm for the projected data (with 4-dimensions).
# 
# (a) Here we implement the EM algorithm. Use the following initialization
# • initialization for mean: random Gaussian vector with zero mean
# • initialization for covariance: generate two Gaussian random matrix of size n-byn:
# S1 and S2, and initialize the covariance matrix for the two components are 
# 
# $$Σ_{1}=S_{1}S_{1}^{T}+I_{n} and Σ_{2}=S_{2}S_{2}^{T}+I_{n}$$
# 
# where $I_{n}$ is an identity matrix of size n-by-n. Plot the log-likelihood function versus the number of iterations to show your algorithm is converging.
# 
# 

# In[11]:


n,m = data.shape

# initialzation: GMM - EM 
k = 2 # number of mixture components

# mixture means init
np.random.seed(20)
mu_old = np.random.randn(k, n)
mu_new = mu_old + 10

# mixture covariance matric init
sigma_old = np.empty((k, n, n))
for ii in range(k):
    tmp = np.random.randn(n, n)
    tmp = tmp@tmp.T +1
    sigma_old[ii] = tmp
sigma_new = sigma_old + 10


# In[12]:


#  prior init
pi_old = np.random.random(k)
pi_old = pi_old/pi_old.sum()
pi_new = np.zeros(2)

# posterior init
tau = np.zeros((m,k), dtype=float)
datamean = data.mean(axis = 1)
ll_all = []
itt = 1
maxIter =35
ntrunc = 5 # truncation lowrank approx. no.
ll_flag = 1000 # see  log-likelihood change w/ iterations


# In[13]:


def lrpdf(data, mu, sigma, k):
    '''
    Calculate the lowrank approximation to the covariance matix
    & the log-likelihood 
    '''
    dim, m = data.shape # numpy.array
    #  dim-by-m dimension 
    if len(mu.shape)==1:
        mu=mu[:, np.newaxis]
    u, s, _ = np.linalg.svd(sigma)
    ut = u[:, 0:k] # select k eigenvectors, ut: dim-by-k
    st = s[0:k] # select k eigenvalues
    dt = np.diag(np.power(st, -1/2)) 
    zx = (data - mu).T @ ut@ dt # projected data, (x-mu)^T * dt
    eterm = - np.sum(zx**2, axis =1) / 2
    # calculate the constant term
    cterm = - np.sum(np.log(st))/2
    logl = eterm + cterm
    return logl


# In[14]:


while itt<maxIter:
    print('Iteration: ',str(itt))
    #  E-step
    for ii in range(k):
        ll_tmp = lrpdf(data, mu_old[ii], sigma_old[ii], ntrunc)
        tau[:, ii] = np.exp(ll_tmp) * pi_old[ii]
    tmp = tau.sum(axis=1)
    tmp = tmp[:, np.newaxis]
    tau = tau / tmp
    # M-step
    # priors: update
    pi_new = tau.sum(axis=0)/m
    # mean: update
    mu_new = tau.T @ data.T
    tmp = tau.sum(axis=0)
    tmp = tmp[:, np.newaxis]
    mu_new = mu_new/tmp
    # cov matrix:update
    for ii in range(k):
        tmp = data - mu_new[ii][:, np.newaxis]
        sigma_new[ii] = tmp @ np.diag(tau[:,ii]) @tmp.T / tau[:,ii].sum()
    
    # log-likelihood, lowrank-approximation
    log_likelihood = 0 
    for ii in range(k):
        ll = lrpdf(data, mu_new[ii], sigma_new[ii], ntrunc) + m*np.log(pi_new[ii])
        log_likelihood = log_likelihood+ ll.sum()
    ll_all.append(log_likelihood)
    mu_old = mu_new.copy()
    sigma_old = sigma_new.copy()
    pi_old = pi_new.copy()
    itt = itt+1
plt.plot(np.array(ll_all)/m)  #converging
plt.title('Itreations & Log-likelihood: 34 Itreations to Converge')


# In[15]:


# ### Viz: mean and covariance matrix
fig2, ax2 = plt.subplots(2, 2)
for ii in range(k):
    im = mu_new[ii].reshape(28, 28)
    ax2[ii, 0].imshow(im.T, cmap='gray')
    ax2[ii, 0].set_title('Mean')
    ax2[ii, 0].set_xticks([])
    ax2[ii, 0].set_yticks([])
    
    ax2[ii, 1].imshow(sigma_new[ii], cmap='gray')
    ax2[ii, 1].set_title('Covariance Matrix')
    ax2[ii, 1].set_xticks([])
    ax2[ii, 1].set_yticks([])


# In[16]:


# Accuracy
idx2 = np.where(label==2)[0]
idx6 = np.where(label==6)[0]

match2 = tau[idx2,0]>=tau[idx2,1]
acc2 = match2.sum()/idx2.size
acc2 = np.max([acc2, 1-acc2])
print('Accuracy 2:', str(acc2))

match6 = tau[idx6,0]>=tau[idx6,1]
acc6 = match6.sum()/idx6.size
acc6 = np.max([acc6, 1-acc6])
print('Accuracy 6:', str(acc6))


# (b) Report of the fitted GMM model when EM has terminated in the algorithms
# as follows. Report of the weights for each component, and the mean of each component,
# by mapping them back to the original space and reformat the vector to make them
# into 28-by-28 matrices and show images. We  see these
# means corresponds to an “average” of the images. We report the two 4-by-4
# covariance matrices by visualizing their intensities (e.g., using a gray scaled image or
# heat map).
# 
# (c) We use the τik
# to infer the labels of the images, and compare with the true labels.
# Report the mis-classification rate for digits “2” and “6” respectively. 
# 
# Here, we perform K-means clustering with K = 2.  We show the mis-classification rate for digits “2” and “6” respectively,
# and compare with GMM. We determine the one achieves the better performance.
# 
# 
# K-means on MNIST dataset
# To compute purity , each cluster is assigned to the class which is most frequent in the cluster, and then the accuracy of this assignment is measured by counting the number of correctly assigned documents and dividing by $N$. Bad clusterings have purity values close to 0, a perfect clustering has a purity of 1

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import math
from numpy.random import choice
from PIL import Image
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
import time
from datetime import timedelta
import matplotlib.pyplot as plt

import scipy.io as sio
import show_image  # library for pandas
from scipy.sparse import csc_matrix, find
from __future__ import division
import random
from scipy.spatial.distance import cdist  
import numexpr as ne
from scipy.spatial import distance

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[18]:


def within_cluster_sum_of_s(S):
    return np.sum(np.amin(S, axis=1))

def kmeans_processor_DDD(X, k, max_steps=np.inf):
    s = choice(len(X), size=k, replace=False) #init_centers: Randomly samples k observations from X as centers
    centers =  X[s, :]   
    converged = False
    labels = np.zeros(len(X))
    i = 1
    while (not converged) and (i <= max_steps):
        old_centers = centers
        
        S= distance.cdist(X, centers, 'cityblock')
        labels = np.argmin(S, axis=1)#assign_cluster_labels(S)
        m, d = X.shape #update_centers
        k = max(labels) + 1
        assert m == len(labels)
        assert (min(labels) >= 0)
        centers = np.empty((k, d))
        for j in range(k): # Compute the new center of cluster j,
            #centers[j, :d] = np.mean(X[labels == j, :], axis=0)  
            centers[j, :d] = np.median(X[labels == j, :], axis=0) #<<<<<<<<<<<<

        converged = set([tuple(x) for x in old_centers]) == set([tuple(x) for x in centers])
        if math.isnan(within_cluster_sum_of_s (S)):
            print ("####empty cluster, lower the number of clusters,k")
            return labels + 0.5
        else:
            i += 1
    print("#####")
    print ("Iterations to converge=", i)
    return labels, centers


# In[19]:


labels_m, centers_m = kmeans_processor_DDD(data_s, 2)


# In[20]:


labels_m.shape, centers_m.shape, data_s.shape


# In[21]:


labels_m


# In[22]:


centers_m


# In[23]:


centroids_r = centers_m.reshape(2,28,28)
centroids = centroids_r * 255
centroids.shape


# In[24]:


plt.imshow(centroids[0].T, cmap='gray')


# In[25]:


plt.imshow(centroids[1].T, cmap='gray')


# In[26]:


def purity_score_processor_m (centers, c):
    unique, counts = np.unique(centers[c], return_counts=True)
    uc = np.asarray((unique, counts)).T
    purity_score_raw = np.amax(uc[:, 1])/np.sum(uc[:, 1])
    purity_score = np.round(purity_score_raw, 3)
    print('Purity score w/ Manhattan Distance for centroid: ', c)
    print(purity_score)
    


# In[27]:


purity_score_all_centroids ={}
for n in [0,1]:
    purity_score_all_centroids[n]=purity_score_processor_m (centers_m, n)

