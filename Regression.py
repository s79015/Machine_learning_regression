#!/usr/bin/env python
# coding: utf-8

# # Regression   
#     

# # Yerzhan Apsattarov 

# Inspired from the LYSTO challenge (https://lysto.grand-challenge.org/LYSTO/), The objective is to develop a regression model for calculating the number of certain type of cells (called lymphocytes) in a given histopathology image patch. For this assignment, all you have to know is that these cells appear in the given image (technically called a immunohistochemistry or IHC image) with a blue nucleus and a brown membrane. Your task is to develop a machine learning model that uses training data (patch images with given cell counts) to predict cell counts in test images.
# The data ‘breast.h5’ can be downloaded from: http://shorturl.at/fuCEO
# The subset of the challenge dataset that you have been given focuses on breast tissue images from a total of 18 different individuals. You can read the data as follows:

# In[1]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.decomposition import PCA
from skimage.color import rgb2hed
from sklearn.metrics import make_scorer
from scipy.stats import entropy

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd

#regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from math import sqrt

from sklearn.neural_network import MLPClassifier

np.seterr(divide='ignore', invalid='ignore') # if code is trying to "divide by zero" or "divide by NaN"


# In[2]:


D = h5py.File('breast.h5', 'r')
X,Y,P = D['images'],np.array(D['counts']),np.array(D['id'])


# In[2]:


print(X,Y,P)
list(D.keys())


# Here, X, Y and P contain the Images, Cell Counts, and Patient IDs, respectively.
# 
# Training and Testing: Use data from patient IDs 1-13 for training and cross validation and 14-18 for testing. Be sure not to test on the images of patients you have used in your training. Each image is in RGB space so it is represented by an array of size 299x299x3 where the first two dimensions correspond to the width and height of the image and the last three correspond to the R,G and B channel.

# # Question No. 1: (Showing data)
# Load the training and test data files and answer the following questions:

# i. How many training and test examples are there?

# In[3]:


#Xtrain
Xtrain=X[P==1,:]

for i in range (2,14):
    Xtrain_2=X[P==i,:]
    Xtrain=np.vstack((Xtrain,Xtrain_2))
#Xtest
Xtest=X[P==14,:]

for i in range (15,19):
    Xtest_2=X[P==i,:]
    Xtest=np.vstack((Xtest,Xtest_2)) 
    
print("Xtrain shape=",Xtrain.shape)
print("Xtest shape=",Xtest.shape)


# In[4]:


#Ytrain
Ytrain=Y[P==1]

for i in range (2,14):
    Ytrain_2=Y[P==i]
    Ytrain=np.hstack((Ytrain,Ytrain_2))

#Ytest
Ytest=Y[P==14]

for i in range (15,19):
    Ytest_2=Y[P==i]
    Ytest=np.hstack((Ytest,Ytest_2)) 
    
print("Ytrain shape=",Ytrain.shape)
print("Ytest shape=",Ytest.shape)


# Answer:
# 
# There are 5841 training and 1563 testing examples.

# ii. Show some image examples using plt.imshow. Describe your observations on what you
# see in the images and how it correlates with the cell count (target variable). []

# In[40]:


import matplotlib.pyplot as plt
for i in range(0,4):
      plt.figure(figsize = (40,10))
      plt.imshow(Xtrain[i])
      plt.show()  
      print("Cell count in the Ytrain =",Ytrain[i])


# Answer:
# 
# As can be seen in the Figures, we can observe that there are blue, brown cells with different intensities. Also there are some arears where brown colour is highly concentrated, but does not have a round shape like cells. According to target variables, we can say that it estimate the brown cells, which clearly expressed. On other hand, I can not see the blue nucleus in these brown cells due to quality of pictures.  

# iii. Plot the histogram of counts. How many images have counts within each of the following
# bins?
# 
# 0 (no lymphocytes)
# 
# 1-5
# 
# 6-10
# 
# 11-20
# 
# 21-50
# 
# 51-200
# 
# >200

# In[32]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20,8))
bins=[0,1,6,11,21,51,200]
plt.hist(Y, bins=bins)
plt.xticks(bins)
plt.xlim(xmin=0, xmax = 70)
plt.title('The histogram of counts',fontsize=15)
plt.show()


# In[37]:


hist, bin_edges = np.histogram(Y, bins=bins)
bins_list=['0 (no lymphocytes)','1-5','6-10','11-20','21-50','51-200']
for i in range (len(bins_list)):
    print(bins_list[i],'have',hist[i],'images')


# Answer:
#    
# 0 (no lymphocytes) have 1397 images
# 
# 1-5 have 4811 images
# 
# 6-10 have 736 images
# 
# 11-20 have 356 images
# 
# 21-50 have 103 images
# 
# 51-200 have 1 image

# iv. Pre-processing: Convert and view a few images from RGB space to HED space and show the
# D channel which should identify the brown elements in the image. For this purpose, you can
# use the color separation notebook available here: https://scikitimage.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html [5
# ]

# In[6]:


import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap

# Create an artificial color close to the original one
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                             'saddlebrown'])

for i in range(0,4):
    print('Image',i)
    ihc_rgb = Xtrain[i]
    ihc_hed = rgb2hed(ihc_rgb)
    
    plt.figure(figsize = (40,10))
    plt.imshow(ihc_hed[:, :, 2],cmap=cmap_dab)
    plt.show() 
    


# v. Do a scatter plot of the average of the brown channel for each image vs. its cell count. Do
# you think this feature would be useful in your regression model? Explain your reasoning. [3
# ]
# 

# In[7]:


import matplotlib.pyplot as plt
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap

brown_channel=[]
for i in range(len(Xtrain)):
    ihc_rgb = Xtrain[i]
    ihc_hed = rgb2hed(ihc_rgb)
    brown_channel.append(np.average(ihc_hed[:, :, 2]))
plt.scatter(brown_channel,Ytrain)  


# Answer:
# 
# As can be seen in the scatter plot, one can notice that these features might be useful in the regression model. Because we can see a positive correlation. However, there are a lot of outliers, especially which starts on a rectangle from (-0.36,0;-0.36,10). These outliers might have a noticeable impact on the regression model line.

# vi. What is the number of images for each patient? Do you think this can have an impact on
# your regression model? []

# In[15]:



sum=0
for i in range(1,19):
    patient=X[P==i,:]
    sum+=len(patient)
    print("The patient with ID",i, "has:", len(patient),"images")
    #print("Sum", sum)


# Answer:
# 
# The patient with ID 1 has: 320 images
# 
# The patient with ID 2 has: 465 images
# 
# The patient with ID 3 has: 958 images
# 
# The patient with ID 4 has: 192 images
# 
# The patient with ID 5 has: 44 images
# 
# The patient with ID 6 has: 105 images
# 
# The patient with ID 7 has: 83 images
# 
# The patient with ID 8 has: 632 images
# 
# The patient with ID 9 has: 533 images
# 
# The patient with ID 10 has: 552 images
# 
# The patient with ID 11 has: 761 images
# 
# The patient with ID 12 has: 791 images
# 
# The patient with ID 13 has: 405 images
# 
# The patient with ID 14 has: 105 images
# 
# The patient with ID 15 has: 399 images
# 
# The patient with ID 16 has: 604 images
# 
# The patient with ID 17 has: 103 images
# 
# The patient with ID 18 has: 352 images
# 
# I believe that the number of images can have an impact on the regression model. Because patients have a different number of images. For example, patient with ID "3" has the largest number of images (958, almost 13% of the dataset), whereas a patient with ID "5" has the lowest number of images (44, 0.6% of the dataset). Because each image contains a different number of cells and target values, the linear regression models created by patient ID will have a distinct amount of samples. 

# vii. What performance metrics can you use for this purpose? Which one will be the best
# performance metric for this problem? Please give reasoning. []
# 

# Anwer:
# Mean Squared Error(MSE):
# MSE or Mean Squared Error is one of the most preferred metrics for regression tasks. It is simply the average of the squared difference between the target value and the value predicted by the regression model. As it squares the differences, it penalizes even a small error which leads to over-estimation of how bad the model is. It is preferred more than other metrics because it is differentiable and hence can be optimized better.
# 
# Root-Mean-Squared-Error(RMSE):
# RMSE is the most widely used metric for regression tasks and is the square root of the averaged squared difference between the target value and the value predicted by the model. It is preferred more in some cases because the errors are first squared before averaging which poses a high penalty on large errors. This implies that RMSE is useful when large errors are undesired.
# 
# Mean-Absolute-Error(MAE):
# MAE is the absolute difference between the target value and the value predicted by the model. The MAE is more robust to outliers and does not penalize the errors as extremely as mse. MAE is a linear score which means all the individual differences are weighted equally. It is not suitable for applications where you want to pay more attention to the outliers.
# 
# R² or Coefficient of Determination:
# Coefficient of Determination or R² is another metric used for evaluating the performance of a regression model. The metric helps us to compare our current model with a constant baseline and tells us how much our model is better. The constant baseline is chosen by taking the mean of the data and drawing a line at the mean. R² is a scale-free score that implies it doesn't matter whether the values are too large or too small, the R² will always be less than or equal to 1.
# 
# Adjusted R²:
# Adjusted R² depicts the same meaning as R² but is an improvement of it. R² suffers from the problem that the scores improve on increasing terms even though the model is not improving which may misguide the researcher. Adjusted R² is always lower than R² as it adjusts for the increasing predictors and only shows improvement if there is a real improvement.
# 
# Answer:
# 
# According to these Linear regression metrics, I believe that RMSE will be best performance metric for this project. The RMSE is a quadratic scoring rule which measures the average magnitude of the error. The equation for the RMSE is given in both of the references. Expressing the formula in words, the difference between forecast and corresponding observed values are each squared and then averaged over the sample. Finally, the square root of the average is taken. Since the errors are squared before they are averaged, the RMSE gives a relatively high weight to large errors. This means the RMSE is most useful when large errors are particularly undesirable.
#     

# # Question No. 2: (Feature Extraction and Classical Regression) []
# 

# # i. Extract features from a given image. Specifically, calculate the:

# a. average of the “brown”, red, green and blue channels

# In[5]:


from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap
#brown
ihc_rgb = X[0]
ihc_hed = rgb2hed(ihc_rgb)
print ("Average of the brown:", np.average(ihc_hed[:, :, 2]))
#red
print ("Average of the red  :", np.average(X[0,:,:,0]))
#green
print ("Average of the green:", np.average(X[0,:,:,1]))
#blue
print ("Average of the blue :", np.average(X[0,:,:,2]))


# Answer (Image=0):
# 
# Average of the brown: -0.3799775017875077
# 
# Average of the red  : 179.0747083365958
# 
# Average of the green: 170.9491169002584
# 
# Average of the blue : 178.14088209304145

# b. variance of the “brown”, red, green and blue channels

# In[10]:


from skimage.color import rgb2hed
#brown
ihc_rgb = X[0]
ihc_hed = rgb2hed(ihc_rgb)
print ("Variance of the brown:", np.var(ihc_hed[:, :, 2]))
#red
print ("Variance of the red  :", np.var(X[0,:,:,0]))
#green
print ("Variance of the green:", np.var(X[0,:,:,1]))
#blue
print ("Variance of the blue :", np.var(X[0,:,:,2]))


# Answer (Image=0):
# 
# Variance of the brown: 0.0005931334288957775
# 
# Variance of the red  : 1921.4295144687405
# 
# Variance of the green: 2212.668085734827
# 
# Variance of the blue : 1596.1474098727995

# c. entropy of the “brown”, red, green and blue channels

# In[7]:


from skimage.color import rgb2hed
from scipy.stats import entropy
import functools
import operator
ihc_rgb = X[0]
ihc_hed = rgb2hed(ihc_rgb)
List_flat_brown=hc_hed[:, :, 2].flatten()

List_flat_red=X[0,:,:,0].flatten()

List_flat_green=X[0,:,:,1].flatten()

List_flat_blue=X[0,:,:,2].flatten()
    
#print("Original List:",hc_hed[:, :, 2])
#print("Flattened List:",List_flat_brown)

print ("Entropy of the brown:", entropy(List_flat_brown))
#red
print ("Entropy of the red  :", entropy(List_flat_red))
#green
print ("Entropy of the green:", entropy(List_flat_green))
#blue
print ("Entropy of the blue :", entropy(List_flat_blue))


# Answer (Image=0):
# 
# Entropy of the brown: 11.398742842865293
# 
# Entropy of the red  : 11.367848794959706
# 
# Entropy of the green: 11.358190087584395
# 
# Entropy of the blue : 11.372237658487851

# d. Histogram of each channel

# In[9]:


import matplotlib.pyplot as plt

#brown
plt.hist(List_flat_brown); plt.title('The histogram of brown chanels',fontsize=15);plt.show()
#red
plt.hist(List_flat_red); plt.title('The histogram of red chanels',fontsize=15);plt.show()
#green
plt.hist(List_flat_green); plt.title('The histogram of green chanels',fontsize=15);plt.show()
#blue
plt.hist(List_flat_blue); plt.title('The histogram of blue chanels',fontsize=15);plt.show()


# e. PCA Coefficients (you may want to use randomized PCA or incremental PCA, see:
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) 

# In[6]:


#red
Red_flat=Xtrain[0,:,:,0]
Red_flat=Red_flat.flatten()
#green
Green_flat=Xtrain[0,:,:,1]
Green_flat=Green_flat.flatten()
#blue
Blue_flat=Xtrain[0,:,:,2]
Blue_flat=Blue_flat.flatten()
#brown
#flattened
ihc_rgb = Xtrain[0]
ihc_hed = rgb2hed(ihc_rgb)
Brown_flat=ihc_hed[:, :, 2]
Brown_flat=Brown_flat.flatten()


# In[57]:


pca = PCA(n_components=350,svd_solver='randomized')
#red
#training PCA
pca.fit(Red_flat)
#projecting the data onto Principal components
projected = pca.transform(Red_flat)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Red ')
plt.grid()
plt.show()


# In[58]:


pca = PCA(n_components=350,svd_solver='randomized')
#red
#training PCA
pca.fit(Red_flat)
#projecting the data onto Principal components
projected = pca.transform(Red_flat)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Red ')
plt.grid()
plt.show()

#green
#training PCA
pca.fit(Green_flat) 
#projecting the data onto Principal components
projected = pca.transform(Green_flat)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Green ')
plt.grid()
plt.show()

#blue
#training PCA
pca.fit(Blue_flat) 
#projecting the data onto Principal components
projected = pca.transform(Blue_flat)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Blue ')
plt.grid()
plt.show()

#brown
#training PCA
pca.fit(Brown_flat) 
#projecting the data onto Principal components
projected = pca.transform(Brown_flat)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Brown ')
plt.grid()
plt.show()


# In[63]:


#red
print("Red channel")
for i in range (340,360):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Red_flat) 
#projecting the data onto Principal components
    projected = pca.transform(Red_flat)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))

print("Green channel")
for i in range (330,340):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Green_flat) 
#projecting the data onto Principal components
    projected = pca.transform(Green_flat)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))   

print("Blue channel")
for i in range (300,320):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Blue_flat) 
#projecting the data onto Principal components
    projected = pca.transform(Blue_flat)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))

print("Brown channel")
for i in range (270,280):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Brown_flat)  
#projecting the data onto Principal components
    projected = pca.transform(Brown_flat)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))


# Answer:
# 
# According to the results, we need to use next PCA dimensions to explain 95% variance of for the given image :
# 
# For red channel at least 351 dimensions.
# 
# For green channel at least 339 dimensions.
# 
# For blue channel at least 313 dimensions.
# 
# For brown channel at least 278 dimensions.

# f. Any other features that you think can be useful for this work. Describe your
# reasoning for using these features. 

# Plot the scatter plot and calculate the correlation coefficient of each feature you obtain
# vs. the target variable (cell count) across all images. Which features do you think are
# important? Give your reasoning. ]
# 

# Answer:

# In order to find the correalation coefficients between each feature you obtain vs. the target variable (cell count) across all images I will use next steps:
# 
# 1) Use the Flatten function for each colour channels across all images in order to apply PCA.
# 
# 2) Find the PCA for each colous, which can explain 95% of all images in the "X" extracted from "breast.h5" file.
# 
# 3) After that the project Using the founded PCA will decrease the dimensions of each colours dataset.
# 
# 4) Next I will find the average, variance and entropy of each image after PCA.
# 
# 5) Finally, it will draw the scatter plot and find the correlation coefficients.

# In[16]:


#red
Red_flat_X=X[0,:,:,0]
Red_flat_X=Red_flat_X.flatten()
for i in range(1,len(X)):
    #red
    X_red=X[i,:,:,0]
    X_red=X_red.flatten()
    Red_flat_X=np.vstack((Red_flat_X,X_red))
print("Red_flat_X",Red_flat_X.shape)    


# In[19]:


#green
Green_flat_X=X[0,:,:,1]
Green_flat_X=Green_flat_X.flatten()

for i in range(1,len(X)):
      #green
    X_green=X[i,:,:,1]
    X_green=X_green.flatten()
    Green_flat_X=np.vstack((Green_flat_X,X_green))
print("Green_flat_X",Green_flat_X.shape)


# In[12]:


#blue
Blue_flat_X=X[0,:,:,2]
Blue_flat_X=Blue_flat_X.flatten()

for i in range(1,len(X)):
        #blue
    X_blue=X[i,:,:,2]
    X_blue=X_blue.flatten()
    Blue_flat_X=np.vstack((Blue_flat_X,X_blue))
print("Blue_flat_X",Blue_flat_X.shape)


# In[15]:


#brown
#flattened
ihc_rgb = X[0]
ihc_hed = rgb2hed(ihc_rgb)
Brown_flat_X=ihc_hed[:, :, 2]
Brown_flat_X=Brown_flat_X.flatten()
for i in range(1,len(X)):
        #brown
    ihc_rgb = X[i]
    ihc_hed = rgb2hed(ihc_rgb)
    X_brown=ihc_hed[:, :, 2]
    X_brown=X_brown.flatten()
    Brown_flat_X=np.vstack((Brown_flat_X,X_brown))
print("Brown_flat_X",Brown_flat_X.shape)


# In[2]:


#read the files

#red
Red_f = h5py.File('Red_flat_X.h5', 'r')
Red_flat_X = Red_f['Red_flat_X']
#green
Green_f = h5py.File('Green_flat_X.h5', 'r')
Green_flat_X = Green_f['Red_flat_X']
#blue
Blue_f = h5py.File('Blue_flat_X.h5', 'r')
Blue_flat_X = Blue_f['Blue_flat_X']
#brown
Brown_f = h5py.File('Brown_flat_X.h5', 'r')
Brown_flat_X = Brown_f['Brown_flat_X'] 


# In[3]:


pca = PCA(n_components=1200,svd_solver='randomized')
#red
Red_f = h5py.File('Red_flat_X.h5', 'r')
Red_flat_X = Red_f['Red_flat_X']
#training PCA
pca.fit(Red_flat_X)
#projecting the data onto Principal components
projected = pca.transform(Red_flat_X)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Red_flat_X ')
plt.grid()
plt.show()

#green
Green_f = h5py.File('Green_flat_X.h5', 'r')
Green_flat_X = Green_f['Red_flat_X']
#training PCA
pca.fit(Green_flat_X) 
#projecting the data onto Principal components
projected = pca.transform(Green_flat_X)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Green_flat_X ')
plt.grid()
plt.show()

#blue
Blue_f = h5py.File('Blue_flat_X.h5', 'r')
Blue_flat_X = Blue_f['Blue_flat_X']
#training PCA
pca.fit(Blue_flat_X) 
#projecting the data onto Principal components
projected = pca.transform(Blue_flat_X)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Blue_flat_X ')
plt.grid()
plt.show()

#brown
Brown_f = h5py.File('Brown_flat_X.h5', 'r')
Brown_flat_X = Brown_f['Brown_flat_X']   
#training PCA
pca.fit(Brown_flat_X) 
#projecting the data onto Principal components
projected = pca.transform(Brown_flat_X)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Brown_flat_X ')
plt.grid()
plt.show()


# In[4]:


#red

print("Red_flat_X channel")
for i in range (1300,1350,10):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Red_flat_X) 
#projecting the data onto Principal components
    projected = pca.transform(Red_flat_X)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))

    #green

print("Green_flat_X channel")
for i in range (1300,1350,10):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Green_flat_X) 
#projecting the data onto Principal components
    projected = pca.transform(Green_flat_X)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))   

    #Blue

print("Blue_flat_X channel")
for i in range (1300,1350,10):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Blue_flat_X) 
#projecting the data onto Principal components
    projected = pca.transform(Blue_flat_X)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))

  #Brown

print("Brown_flat_X channel")
for i in range (900,1000,10):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Brown_flat_X)  
#projecting the data onto Principal components
    projected = pca.transform(Brown_flat_X)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))


# According to the results, we need to use next PCA dimensions to explain about 95% variance of all images in the dateset:
# 
# For red channel  around 1400 dimensions.
# 
# For green channel around 1400  dimensions.
# 
# For blue channel  around 1400  dimensions.
# 
# For brown channel around 1100  dimensions.

# In[5]:


#red
pca= PCA(n_components=1400,svd_solver='randomized')
#training PCA
pca.fit(Red_flat_X) 
#projecting the data onto Principal components
Red_flat_X_pca = pca.transform(Red_flat_X)
    
             #green
pca= PCA(n_components=1400,svd_solver='randomized')
#training PCA
pca.fit(Green_flat_X) 
#projecting the data onto Principal components
Green_flat_X_pca = pca.transform(Green_flat_X)

             #blue
pca= PCA(n_components=1400,svd_solver='randomized')
#training PCA
pca.fit(Blue_flat_X) 
#projecting the data onto Principal components
Blue_flat_X_pca = pca.transform(Blue_flat_X)

            #brown
pca= PCA(n_components=1100,svd_solver='randomized')
#training PCA
pca.fit(Brown_flat_X) 
#projecting the data onto Principal components
Brown_flat_X_pca = pca.transform(Brown_flat_X)

  


# In[6]:


print("Red_flat_X_pca",Red_flat_X_pca.shape)
print("Green_flat_X_pca",Green_flat_X_pca.shape)
print("Blue_flat_X_pca",Blue_flat_X_pca.shape)
print("Brown_flat_X_pca",Brown_flat_X_pca.shape)  


# In[8]:


#red
Red_flat_X_pca_var=np.var(Red_flat_X_pca[0,])
Red_flat_X_pca_avg=np.mean(Red_flat_X_pca[0,])
Red_flat_X_pca_ent=entropy(Red_flat_X_pca[0,])
for i in range(1,len(Red_flat_X_pca)):
   #var
    red_var=np.var(Red_flat_X_pca[i,])
    Red_flat_X_pca_var=np.vstack((Red_flat_X_pca_var,red_var))
   #average
    red_avg=np.mean(Red_flat_X_pca[i,])
    Red_flat_X_pca_avg=np.vstack((Red_flat_X_pca_avg,red_avg))  
   #entropy
    red_ent=entropy(Red_flat_X_pca[i,])
    Red_flat_X_pca_ent=np.vstack((Red_flat_X_pca_ent,red_ent))

#green
Green_flat_X_pca_var=np.var(Green_flat_X_pca[0,])
Green_flat_X_pca_avg=np.mean(Green_flat_X_pca[0,])
Green_flat_X_pca_ent=entropy(Green_flat_X_pca[0,])
for i in range(1,len(Green_flat_X_pca)):
   #var
    green_var=np.var(Green_flat_X_pca[i,])
    Green_flat_X_pca_var=np.vstack((Green_flat_X_pca_var,green_var))
   #average
    green_avg=np.mean(Green_flat_X_pca[i,])
    Green_flat_X_pca_avg=np.vstack((Green_flat_X_pca_avg,green_avg))
   #entropy
    green_ent=entropy(Green_flat_X_pca[i,])
    Green_flat_X_pca_ent=np.vstack((Green_flat_X_pca_ent,green_ent))
    
#blue
Blue_flat_X_pca_var=np.var(Blue_flat_X_pca[0,])
Blue_flat_X_pca_avg=np.mean(Blue_flat_X_pca[0,])
Blue_flat_X_pca_ent=entropy(Blue_flat_X_pca[0,])
for i in range(1,len(Blue_flat_X_pca)):
   #var
    blue_var=np.var(Blue_flat_X_pca[i,])
    Blue_flat_X_pca_var=np.vstack((Blue_flat_X_pca_var,blue_var))
   #average
    blue_avg=np.mean(Blue_flat_X_pca[i,])
    Blue_flat_X_pca_avg=np.vstack((Blue_flat_X_pca_avg,blue_avg)) 
   #entropy
    blue_ent=entropy(Blue_flat_X_pca[i,])
    Blue_flat_X_pca_ent=np.vstack((Blue_flat_X_pca_ent,blue_ent))
    
    #brown
Brown_flat_X_pca_var=np.var(Brown_flat_X_pca[0,])
Brown_flat_X_pca_avg=np.mean(Brown_flat_X_pca[0,])
Brown_flat_X_pca_ent=entropy(Brown_flat_X_pca[0,])
for i in range(1,len(Brown_flat_X_pca)):
   #var
    brown_var=np.var(Brown_flat_X_pca[i,])
    Brown_flat_X_pca_var=np.vstack((Brown_flat_X_pca_var,brown_var))
   #average
    brown_avg=np.mean(Brown_flat_X_pca[i,])
    Brown_flat_X_pca_avg=np.vstack((Brown_flat_X_pca_avg,brown_avg)) 
    #entropy
    brown_ent=entropy(Brown_flat_X_pca[i,])
    Brown_flat_X_pca_ent=np.vstack((Brown_flat_X_pca_ent,brown_ent))


# In[12]:


#red
Red_flat_X_pca_var=Red_flat_X_pca_var[:,0]
Red_flat_X_pca_avg=Red_flat_X_pca_avg[:,0]
Red_flat_X_pca_ent=Red_flat_X_pca_ent[:,0]
print("Red_flat_X_pca_var",Red_flat_X_pca_var[:5])
print("Red_flat_X_pca_avg",Red_flat_X_pca_avg[:5])
print("Red_flat_X_pca_ent",Red_flat_X_pca_ent[:5])

#green
Green_flat_X_pca_var=Green_flat_X_pca_var[:,0]
Green_flat_X_pca_avg=Green_flat_X_pca_avg[:,0]
Green_flat_X_pca_ent=Green_flat_X_pca_ent[:,0]
print("Green_flat_X_pca_var",Green_flat_X_pca_var[:5])
print("Green_flat_X_pca_avg",Green_flat_X_pca_avg[:5])
print("Green_flat_X_pca_ent",Green_flat_X_pca_ent[:5])

#blue
Blue_flat_X_pca_var=Blue_flat_X_pca_var[:,0]
Blue_flat_X_pca_avg=Blue_flat_X_pca_avg[:,0]
Blue_flat_X_pca_ent=Blue_flat_X_pca_ent[:,0]
print("Blue_flat_X_pca_var",Blue_flat_X_pca_var[:5])
print("Blue_flat_X_pca_avg",Blue_flat_X_pca_avg[:5])
print("Blue_flat_X_pca_ent",Blue_flat_X_pca_ent[:5])

#brown
Brown_flat_X_pca_var=Brown_flat_X_pca_var[:,0]
Brown_flat_X_pca_avg=Brown_flat_X_pca_avg[:,0]
Brown_flat_X_pca_ent=Brown_flat_X_pca_ent[:,0]
print("Brown_flat_X_pca_var",Brown_flat_X_pca_var[:5])
print("Brown_flat_X_pca_avg",Brown_flat_X_pca_avg[:5])
print("Brown_flat_X_pca_ent",Brown_flat_X_pca_ent[:5])


# In[13]:


#correlations
print("Correlation coefficients")
           #red
Red_flat_X_pca_var_Y_cor, _ = np.corrcoef(Red_flat_X_pca_var,Y) 
Red_flat_X_pca_avg_Y_cor, _ = np.corrcoef(Red_flat_X_pca_avg,Y)
Red_flat_X_pca_ent_Y_cor, _ = np.corrcoef(Red_flat_X_pca_ent,Y)
print("Red")
print("All images variance and all target:",Red_flat_X_pca_var_Y_cor)
print("All images average  and all target:",Red_flat_X_pca_avg_Y_cor)
print("All images entropy  and all target:",Red_flat_X_pca_ent_Y_cor)

            #green
Green_flat_X_pca_var_Y_cor, _ = np.corrcoef(Green_flat_X_pca_var,Y) 
Green_flat_X_pca_avg_Y_cor, _ = np.corrcoef(Green_flat_X_pca_avg,Y)
Green_flat_X_pca_ent_Y_cor, _ = np.corrcoef(Green_flat_X_pca_ent,Y)
print("Green")
print("All images variance and all target:",Green_flat_X_pca_var_Y_cor)
print("All images average  and all target:",Green_flat_X_pca_avg_Y_cor)
print("All images entropy  and all target:",Green_flat_X_pca_ent_Y_cor)

            #blue
Blue_flat_X_pca_var_Y_cor, _ = np.corrcoef(Blue_flat_X_pca_var,Y) 
Blue_flat_X_pca_avg_Y_cor, _ = np.corrcoef(Blue_flat_X_pca_avg,Y)
Blue_flat_X_pca_ent_Y_cor, _ = np.corrcoef(Blue_flat_X_pca_ent,Y)
print("Blue")
print("All images variance and all target:",Blue_flat_X_pca_var_Y_cor)
print("All images average  and all target:",Blue_flat_X_pca_avg_Y_cor)
print("All images entropy  and all target:",Blue_flat_X_pca_ent_Y_cor)

             #brown
Brown_flat_X_pca_var_Y_cor, _ = np.corrcoef(Brown_flat_X_pca_var,Y) 
Brown_flat_X_pca_avg_Y_cor, _ = np.corrcoef(Brown_flat_X_pca_avg,Y)
Brown_flat_X_pca_ent_Y_cor, _ = np.corrcoef(Brown_flat_X_pca_ent,Y)
print("Brown")
print("All images variance and all target:",Brown_flat_X_pca_var_Y_cor)
print("All images average  and all target:",Brown_flat_X_pca_avg_Y_cor)
print("All images entropy  and all target:",Brown_flat_X_pca_ent_Y_cor)


# In[14]:


#red
#average
plt.scatter(Red_flat_X_pca_avg, Y);plt.xlabel('Average');plt.ylabel('Cell Count');
plt.title('Average Red and Cell count.')
plt.show()
#variance
plt.scatter(Red_flat_X_pca_var, Y);
plt.xlabel('Variance');plt.ylabel('Cell Count');
plt.title('Variance Red and Cell count.')
plt.show()

               #green 
#average
plt.scatter(Green_flat_X_pca_avg, Y);plt.xlabel('Average');plt.ylabel('Cell Count');
plt.title('Average Green and Cell count.')
plt.show()
#variance
plt.scatter(Green_flat_X_pca_var, Y);
plt.xlabel('Variance');plt.ylabel('Cell Count');
plt.title('Variance Green and Cell count')
plt.show()

                #blue
#average
plt.scatter(Blue_flat_X_pca_avg, Y);plt.xlabel('Average');plt.ylabel('Cell Count');
plt.title('Average Blue and Cell count')
plt.show()
#variance
plt.scatter(Blue_flat_X_pca_var, Y);
plt.xlabel('Variance');plt.ylabel('Cell Count');
plt.title('Variance Blue and Cell count')
plt.show()


                   #brown
#average
plt.scatter(Brown_flat_X_pca_avg, Y);plt.xlabel('Average');plt.ylabel('Cell Count');
plt.title('Average Brown and Cell count')
plt.show()
#variance
plt.scatter(Brown_flat_X_pca_var, Y);
plt.xlabel('Variance');plt.ylabel('Cell Count');
plt.title('Variance Brown and Cell count')
plt.show()


# Answer:
# 
# Correlation coefficients
# 
# Red
# 
# All images variance and all target: 0.22485453
# 
# All images average  and all target: 0.22273208
# 
# All images entropy  and all target: nan
# 
# Green
# 
# All images variance and all target: 0.34428821
# 
# All images average  and all target: 0.25065003
# 
# All images entropy  and all target: nan
# 
# Blue
# 
# All images variance and all target: 0.41304379
# 
# All images average  and all target: 0.2856131
# 
# All images entropy  and all target: nan
# 
# Brown
# 
# All images variance and all target: 0.50792776
# 
# All images average  and all target: 0.31056347
# 
# All images entropy  and all target: nan
# 
# 
# According to results of all channels except "Red channel" show noticeable correlation coefficients in the "Variances" rather than "Averages" of these channels. Considering the "Variances", one can conclude that the Variance of Brown channel represents the best correlation coefficient than other channels. Therefore for regression models, I will use the Variance of Brown channel or brown channel after PCA (which explain 95%) to build the regression models and convolutional neural networks.

# # ii. Try the following regression models with the features used in part-I. You can do 3-fold cross-validation analysis (https://scikit-learn.org/stable/modules/cross_validation.html) to select feature combinations and optimal hyper-parameters for your models. Report your results on the test set by plotting the scatter plot between true and predicted counts for each type of regression model. Also, report your results in terms of RMSE, Correlation Coefficient and R2 score (https://scikitlearn.org/stable/modules/classes.html#module-sklearn.metrics). []

# In order to build the regression models I will find brown channel variances for testing and training. The methods will be the same as in a question 1.

# # XTEST

# In[32]:


#Xtest brown from 4 dim to 2 dim
ihc_rgb = Xtest[0]
ihc_hed = rgb2hed(ihc_rgb)
Brown_flat_test=ihc_hed[:, :, 2]
Brown_flat_test=Brown_flat_test.flatten()
for i in range(1,len(Xtest)):
     #brown
    ihc_rgb = Xtest[i]
    ihc_hed = rgb2hed(ihc_rgb)
    X_brown=ihc_hed[:, :, 2]
    X_brown=X_brown.flatten()
    Brown_flat_test=np.vstack((Brown_flat_test,X_brown))
print("Brown_flat_test",Brown_flat_test.shape)


# In[18]:


Brown_t = h5py.File('Brown_flat_test.h5', 'r')
#Brown_t.keys()
Brown_flat_test = Brown_t['Brown_flat_test ']


# In[34]:


#PCA for training dataset
pca= PCA(n_components=1000,svd_solver='randomized')
#brown
#training PCA
pca.fit(Brown_flat_test ) 
#projecting the data onto Principal components
projected = pca.transform(Brown_flat_test)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Brown XTest')
plt.grid()
plt.show()


# In[36]:


print("PCA for Xtest")
for i in range (450,580,30):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Brown_flat_test)  
#projecting the data onto Principal components
    projected = pca.transform(Brown_flat_test)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))


# In[19]:


pca= PCA(n_components=500,svd_solver='randomized')
#training PCA
pca.fit(Brown_flat_test)  
#projecting the data onto Principal components
Brown_flat_test_pca = pca.transform(Brown_flat_test)
print(Brown_flat_test_pca.shape)


# In[35]:


Brown_flat_test_pca_var=np.var(Brown_flat_test_pca[0,])
Brown_flat_test_pca_avg=np.mean(Brown_flat_test_pca[0,])
for i in range(1,len(Brown_flat_test_pca)):
        #var
    brown_var=np.var(Brown_flat_test_pca[i,])
    Brown_flat_test_pca_var=np.vstack((Brown_flat_test_pca_var,brown_var))
        #average
    brown_avg=np.mean(Brown_flat_test_pca[i,])
    Brown_flat_test_pca_avg=np.vstack((Brown_flat_test_pca_avg,brown_avg))  

#Brown_flat_test_pca_var=Brown_flat_test_pca_var[:,0]
#Brown_flat_test_pca_avg=Brown_flat_test_pca_avg[:,0]
print(Brown_flat_test_pca_var[:5])


# In[36]:


with h5py.File('Brown_flat_test_pca_var.h5', 'w') as Brown_t_pca:
    Brown_t_pca.create_dataset("Brown_flat_test_pca_var",  data=Brown_flat_test_pca_var)


# In[24]:


brown_test_var_ytest_cor, _ = np.corrcoef(Brown_flat_test_pca_var,Ytest) 
brown_test_avg_ytest_cor, _ = np.corrcoef(Brown_flat_test_pca_avg,Ytest)
print("Xtest. Correlation coefficient. Variance and Ytest",brown_test_var_ytest_cor)
print("Xtest. Correlation coefficient. Average and Ytest",brown_test_avg_ytest_cor)


# # XTRAIN

# In[77]:


#brown
#flattened
ihc_rgb = Xtrain[0]
ihc_hed = rgb2hed(ihc_rgb)
Brown_flat_train=ihc_hed[:, :, 2]
Brown_flat_train=Brown_flat_train.flatten()
for i in range(1,len(Xtrain)):
     #brown
    ihc_rgb = Xtrain[i]
    ihc_hed = rgb2hed(ihc_rgb)
    X_brown=ihc_hed[:, :, 2]
    X_brown=X_brown.flatten()
    Brown_flat_train=np.vstack((Brown_flat_train,X_brown))
print("Brown_flat_train",Brown_flat_train.shape)


# In[25]:


Brown_tr = h5py.File('Brown_flat_train.h5', 'r')
Brown_flat_train = Brown_tr['Brown_flat_train ']


# In[9]:


pca= PCA(n_components=1000,svd_solver='randomized')
#brown
#training PCA
pca.fit(Brown_flat_train ) 
#projecting the data onto Principal components
projected = pca.transform(Brown_flat_train)
#plot results
plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph Brown Train')
plt.grid()
plt.show()


# In[11]:


for i in range (1000,1500,100):
    pca= PCA(n_components=i,svd_solver='randomized')
    #training PCA
    pca.fit(Brown_flat_train)  
#projecting the data onto Principal components
    projected = pca.transform(Brown_flat_train)
    
    print('PCA=',i, 'Cum. exp. variance=', sum(pca.explained_variance_ratio_))


# Accordind to results we need to use at least 1200 dimensions to explain more that 95% of "Training dataset images" .

# In[26]:


pca= PCA(n_components=1200,svd_solver='randomized')
#training PCA
pca.fit(Brown_flat_train)  
#projecting the data onto Principal components
Brown_flat_train_pca = pca.transform(Brown_flat_train)
print(Brown_flat_train_pca.shape)


# In[31]:


Brown_flat_train_pca_var=np.var(Brown_flat_train_pca[0,])
                     
for i in range(1,len(Brown_flat_train_pca)):
        #var
    brown_var=np.var(Brown_flat_train_pca[i,])
    Brown_flat_train_pca_var=np.vstack((Brown_flat_train_pca_var,brown_var))

#Brown_flat_train_pca_var=Brown_flat_train_pca_var[:,0]


# In[32]:


#save Brown_flat_train_pca_var
with h5py.File('Brown_flat_train_pca_var.h5', 'w') as Brown_tr_var:
    Brown_tr_var.create_dataset("Brown_flat_train_pca_var",  data=Brown_flat_train_pca_var)


# In[17]:


brown_train_var_ytrain_cor, _ = np.corrcoef(Brown_flat_train_pca_var,Ytrain) 

print("Xtrain. Correlation coefficient. Variance and Ytrain",brown_train_var_ytrain_cor)


# # a. Ordinary Least Squares (OLS) regression

# In[6]:


#Brown_flat_train_pca_var
Brown_tr = h5py.File('Brown_flat_train_pca_var.h5', 'r')
#Brown_tr.keys()
Brown_flat_train_pca_var = Brown_tr['Brown_flat_train_pca_var']

#Brown_flat_test_pca_var
Brown_t = h5py.File('Brown_flat_test_pca_var.h5', 'r')
#Brown_t.keys()
Brown_flat_test_pca_var = Brown_t['Brown_flat_test_pca_var']


# In[23]:


#Grid search with brown variance
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False],'n_jobs':list(range(1,10)) }

#stratified cross-validation and regression
skf = StratifiedKFold(n_splits=3)
regr = LinearRegression()
    #R2
grid = GridSearchCV(regr, parameters, cv=skf,scoring='r2')
grid.fit(Brown_flat_train_pca_var, Ytrain)
print("The best OLS 'Brown variance' with next parametres is: ", grid.best_estimator_, "and best score of 'R2'",grid.best_score_)
    #MSE
grid = GridSearchCV(regr, parameters, cv=skf,scoring='neg_mean_squared_error')
grid.fit(Brown_flat_train_pca_var, Ytrain)
print("The best OLS 'Brown variance' with next parametres is: ", grid.best_estimator_, "and best score of 'MSE'",grid.best_score_)
   


# In[27]:


lm = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
model = lm.fit(Brown_flat_train_pca_var, Ytrain)
predictions = lm.predict(Brown_flat_test_pca_var)
predictions=predictions[:,0]
## The line / model
RMSE = sqrt(mean_squared_error(Ytest, predictions))

Corr = np.corrcoef(Ytest, predictions)
Corr = Corr[1][0]

R2   = r2_score(Ytest, predictions)

plt.scatter(Ytest, predictions)
plt.title('OLS.\n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # b. Multilayer Perceptron (in Keras or PyTorch). 

# In[49]:


import torch
from torch.autograd import Variable
import torch.nn as nn

#Ytrain
Ytrain=Y[P==1]

for i in range (2,14):
    Ytrain_2=Y[P==i]
    Ytrain=np.hstack((Ytrain,Ytrain_2))

#Ytest
Ytest=Y[P==14]

for i in range (15,19):
    Ytest_2=Y[P==i]
    Ytest=np.hstack((Ytest,Ytest_2)) 
    
print("Ytrain shape=",Ytrain.shape)
print("Ytest shape=",Ytest.shape)

Brown_tr_pca = h5py.File('Brown_flat_train_pca_var.h5', 'r')
Brown_flat_train_pca_var = Brown_tr_pca['Brown_flat_train_pca_var']

Brown_te_pca = h5py.File('Brown_flat_test_pca_var.h5', 'r')
Brown_flat_test_pca_var = Brown_te_pca['Brown_flat_test_pca_var']

Xtrain = np.array(Brown_flat_train_pca_var, dtype='float32')
Xtrain = Xtrain.reshape(-1,1)
Xtrain_tensor = Variable(torch.from_numpy(Xtrain))

Ytrain=np.array(Ytrain,dtype='float32')
Ytrain=Ytrain.reshape(-1,1)
Ytrain_tensor = Variable(torch.from_numpy(Ytrain))

Xtest = np.array(Brown_flat_test_pca_var, dtype='float32')
Xtest = Xtest.reshape(-1,1)
Xtest_tensor = Variable(torch.from_numpy(Xtest))

Ytest=np.array(Ytest,dtype='float32')
Ytest=Ytest.reshape(-1,1)
Ytest_tensor = Variable(torch.from_numpy(Ytest))


# In[36]:


from sklearn.metrics import r2_score
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size):
    # super function inherits from nn.Module so that we can access everything from nn.Module
        super(LinearRegression,self).__init__()
        # Linear function
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim)

mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#a plot for the loss function on our model
MSE_list = []
iteration_number = 10001
for iteration in range(iteration_number):

    # perform optimization with zero gradient
    optimizer.zero_grad()

    results = model(Xtrain_tensor)
    loss = mse(results, Ytrain_tensor)
    RMSE = sqrt(mse(results, Ytrain_tensor))
    
    
    results_np=results.detach().numpy()
    Ytrain_tensor_np=np.array(Ytrain_tensor,dtype='float32')
    
    results_np=results_np[:,0]
    Ytrain_tensor_np=Ytrain_tensor_np[:,0]
    #print("results_np.shape",results_np.shape)
    #print("Ytrain_tensor_np",Ytrain_tensor_np.shape)
    Corr = np.corrcoef(results_np, Ytrain_tensor_np)
    Corr = Corr[1][0]
    R2   = r2_score(results_np, Ytrain_tensor_np)
    # calculate derivative by stepping backward
    loss.backward()

    # Updating parameters
    optimizer.step()

    # store loss
    MSE_list.append(loss.data)

    # print loss
    if(iteration % 200 == 0):
        print('epoch= {}, MSE= {}, RMSE= {}, Corr= {},R2= {})'.format(iteration, loss.data,RMSE,Corr,R2))#Corr.data,R2.data))

plt.plot(range(iteration_number),MSE_list)
plt.xlabel("Number of Iterations")
plt.ylabel("MSE")
plt


# According to resluts we can get multilayer perceptron with epoch '10 000' next results:
# 
# MSE= 16.7790584564209, RMSE= 4.096224903056581, Corr= 0.4400096892463234
# 

# In[38]:


predicted = model(Xtest_tensor).data.numpy()

predicted=predicted[:,0]
Ytest=Ytest[:,0]
RMSE = sqrt(mean_squared_error(Ytest, predicted))
Corr = np.corrcoef(Ytest, predicted)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted)
plt.title('MLP. Torch.\n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # c. Ridge Regression (Required For MSc. Students only)

# In[16]:


from sklearn.linear_model import Ridge

Ridge_GS = Ridge()
#print(Ridge.get_params().keys())
parameters = {'alpha':[1000,100,10,1,0.1,0.01,0.001,0.0001,0],"fit_intercept":[True,False]}  

grid_search = GridSearchCV(estimator = Ridge_GS, param_grid = parameters, scoring = 'neg_mean_squared_error', cv = 3)
grid_search = grid_search.fit(Xtrain, Ytrain)
print("The best Ridge regression 'Brown variance' with next parametres is: ", grid_search.best_estimator_, "and best score of 'MSE'",grid_search.best_score_)
   


# In[26]:


from sklearn.linear_model import Ridge

Ridge_model = Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None, normalize=False,random_state=None, solver='auto', tol=0.001)
Ridge_model.fit(Xtrain,Ytrain)
predicted = Ridge_model.predict(Xtest)

predicted=predicted[:,0]
Ytest=Ytest[:,0]

RMSE = sqrt(mean_squared_error(Ytest, predicted))
Corr = np.corrcoef(Ytest, predicted)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted)
plt.title('Ridge Regression. \n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # d. Support Vector Regression (Required For MSc. Students only)

# In[13]:


#Brown_flat_train_pca_var
Brown_tr = h5py.File('Brown_flat_train_pca_var.h5', 'r')
#Brown_tr.keys()
Xtrain = Brown_tr['Brown_flat_train_pca_var']

#Brown_flat_test_pca_var
Brown_t = h5py.File('Brown_flat_test_pca_var.h5', 'r')
#Brown_t.keys()
Xtest = Brown_t['Brown_flat_test_pca_var']


# In[16]:


from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


sc_Xtrain = StandardScaler()
sc_Xtest = StandardScaler()
Xtrain = sc_Xtrain.fit_transform(Xtrain)
Xtest = sc_Xtest.fit_transform(Xtest)

# Grid search

parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}
svr = SVR()
grid_search = GridSearchCV(svr, parameters,cv=3,scoring = 'neg_mean_squared_error',)
grid_search.fit(Xtrain,Ytrain)
print("The best SVR 'Brown variance' with next parametres is: ", grid_search.best_estimator_, "and best score of 'MSE'",grid_search.best_score_)


# In[17]:


#model
model_svr = SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.5, gamma=0.0001,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
model_svr.fit(Xtrain,Ytrain)
predicted = model_svr.predict(Xtest)

#predicted=predicted[:,0]
#Ytest=Ytest[:,0]

RMSE = sqrt(mean_squared_error(Ytest, predicted))
Corr = np.corrcoef(Ytest, predicted)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted)

plt.scatter(Ytest, predicted)
plt.title('Support Vector Regression. \n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# In[12]:


from IPython.display import HTML, display
import tabulate
table = [["" ,"Ordinary Least Squares regression ","Multilayer Perceptron (PyTorch) regression","Ridge Regression","Support Vector Regression"],
        ['RMSE',3.865,3.79,3.604,4.99],
        ['Correlation',0.748,0.748,0.748,0.748],
        ['R2',0.430,0.45,0.505,0.05],
        ]
display(HTML(tabulate.tabulate(table, tablefmt='html')))


# Answer :
# 
# According to the results of 4 regression models, one can conclude that Ridge regression model shows the lowest RMSE value with 3.60 and good R2 than others. After that the second best result was represented by Multilayer Perceptron  (Pytorch), where the RMSE is equal to 3.79 and R2=0.45.

# # Question No. 3 (Using Convolutional Neural Networks) []
# 

# Use a convolutional neural network (in Keras or PyTorch) to solve this problem by directly in much
# the same was as in part (ii) of Question (ii). You are to develop an architecture of the neural network
# that takes an image directly as input and produces a count as the output. You are free to choose any
# network structure as long as you can show that it gives good cross-validation performance. Report
# your results on the test set by plotting the scatter plot between true and predicted counts for each
# type of regression model. Also, report your results in terms of RMSE, Correlation Coefficient and R2 
# score. You will be evaluated on the design on your machine learning model, cross-validation and
# final performance metrics. Try to get the best test performance you can.
# 

# Based on your models, you may want to participate in the challenge as well and report your
# challenge scores (optional but will you can get a bonus if your rank high in the challenge).

# # Keras (brown channel after PCA)

# In[47]:


Brown_tr_pca = h5py.File('Brown_flat_train_pca.h5', 'r')
Brown_flat_train_pca = Brown_tr_pca['Brown_flat_train_pca']

Brown_te_pca = h5py.File('Brown_flat_test_pca.h5', 'r')
#Brown_te_pca.keys()
Brown_flat_test_pca = Brown_te_pca['Brown_flat_test_pca'] 
Ytrain=Ytrain.reshape(-1,1)


# In[45]:


Brown_tr = h5py.File('Brown_flat_train.h5', 'r')
#Brown_tr.keys()
Brown_flat_train = Brown_tr['Brown_flat_train ']

Brown_te = h5py.File('Brown_flat_test.h5', 'r')
#Brown_te.keys()
Brown_flat_test = Brown_te['Brown_flat_test '] 


# In[58]:


pca= PCA(n_components=1200,svd_solver='randomized')
#training PCA
pca.fit(Brown_flat_test)  
#projecting the data onto Principal components
Brown_flat_test_1200 = pca.transform(Brown_flat_test)


# In[48]:


Xtrain=Brown_flat_train_pca
Xtest=Brown_flat_test_pca


# In[21]:


from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.metrics import mean_squared_error


# In[51]:


model = Sequential()
model.add(Dense(2400, input_dim=1200, kernel_initializer='normal', activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


# In[55]:


history=model.fit(Xtrain, Ytrain, epochs=15, batch_size=50,  verbose=1, validation_split=0.2)


# In[56]:


print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[62]:


predicted_test = model.predict(Brown_flat_test_1200)

predicted_test=predicted_test[:,0]
#Ytest=Ytest[:,0]
RMSE = sqrt(mean_squared_error(Ytest, predicted_test))
Corr = np.corrcoef(Ytest, predicted_test)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted_test)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted_test)
plt.title('Convolutional neural network. Keras.\n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # Pytorch (brown channel after PCA)

# In[65]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
torch.manual_seed(1)


# In[64]:


Xtrain = np.array(Brown_flat_train_pca, dtype='float32')
Xtrain_tensor = Variable(torch.from_numpy(Xtrain))

Ytrain=np.array(Ytrain,dtype='float32')
Ytrain=Ytrain.reshape(-1,1)
Ytrain_tensor = Variable(torch.from_numpy(Ytrain))

Xtest = np.array(Brown_flat_test_1200, dtype='float32')
Xtest_tensor = Variable(torch.from_numpy(Xtest))

Ytest=np.array(Ytest,dtype='float32')
Ytest=Ytest.reshape(-1,1)
Ytest_tensor = Variable(torch.from_numpy(Ytest))


# In[66]:


class CNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(CNN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


# In[67]:


cnn = CNN(n_feature=1200, n_hidden=20, n_output=1)     # define the network
print(cnn) 


# In[70]:


# Optimization (find parameters that minimize error)
learning_rate = 0.02
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
#a plot for the loss function on our model
MSE_list = []
iteration_number = 10001
mse = torch.nn.MSELoss()
for iteration in range(iteration_number):

    # perform optimization with zero gradient
    optimizer.zero_grad()

    results = cnn(Xtrain_tensor)
    MSE = mse(results, Ytrain_tensor)
    RMSE = sqrt(mse(results, Ytrain_tensor))
    
    
    results_np=results.detach().numpy()
    Ytrain_tensor_np=np.array(Ytrain_tensor,dtype='float32')
    
    results_np=results_np[:,0]
    Ytrain_tensor_np=Ytrain_tensor_np[:,0]
    #print("results_np.shape",results_np.shape)
    #print("Ytrain_tensor_np",Ytrain_tensor_np.shape)
    Corr = np.corrcoef(results_np, Ytrain_tensor_np)
    Corr = Corr[1][0]
    R2   = r2_score(results_np, Ytrain_tensor_np)
    # calculate derivative by stepping backward
    MSE.backward()

    # Updating parameters
    optimizer.step()

    # store loss
    MSE_list.append(MSE)

    # print loss
    if(iteration % 200 == 0):
        print('epoch= {}, MSE= {}, RMSE= {}, Corr= {},R2= {})'.format(iteration, MSE,RMSE,Corr,R2))#Corr.data,R2.data))

plt.plot(range(iteration_number),MSE_list)
plt.xlabel("Number of Iterations")
plt.ylabel("MSE")
plt


# In[71]:


predicted_test = cnn(Xtest_tensor).data.numpy()

predicted_test=predicted_test[:,0]
Ytest=Ytest[:,0]
RMSE = sqrt(mean_squared_error(Ytest, predicted_test))
Corr = np.corrcoef(Ytest, predicted_test)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted_test)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted_test)
plt.title('Convolutional neural network. Torch.\n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # Another network builded by varainces of brown channel

# In[17]:


Brown_tr_pca = h5py.File('Brown_flat_train_pca_var.h5', 'r')
Brown_flat_train_pca_var = Brown_tr_pca['Brown_flat_train_pca_var']

Brown_te_pca = h5py.File('Brown_flat_test_pca_var.h5', 'r')
Brown_flat_test_pca_var = Brown_te_pca['Brown_flat_test_pca_var']

Xtrain = np.array(Brown_flat_train_pca_var, dtype='float32')
Xtrain = Xtrain.reshape(-1,1)
Xtrain_tensor = Variable(torch.from_numpy(Xtrain))

Ytrain=np.array(Ytrain,dtype='float32')
Ytrain=Ytrain.reshape(-1,1)
Ytrain_tensor = Variable(torch.from_numpy(Ytrain))

Xtest = np.array(Brown_flat_test_pca_var, dtype='float32')
Xtest = Xtest.reshape(-1,1)
Xtest_tensor = Variable(torch.from_numpy(Xtest))

Ytest=np.array(Ytest,dtype='float32')
Ytest=Ytest.reshape(-1,1)
Ytest_tensor = Variable(torch.from_numpy(Ytest))


# In[9]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
torch.manual_seed(1)


# In[89]:


class CNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(CNN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


# In[90]:


cnn = CNN(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(cnn)  # net architecture


# In[91]:


# Optimization (find parameters that minimize error)
learning_rate = 0.02
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
#a plot for the loss function on our model
MSE_list = []
iteration_number = 10001
for iteration in range(iteration_number):

    # perform optimization with zero gradient
    optimizer.zero_grad()

    results = cnn(Xtrain_tensor)
    MSE = mse(results, Ytrain_tensor)
    RMSE = sqrt(mse(results, Ytrain_tensor))
    
    
    results_np=results.detach().numpy()
    Ytrain_tensor_np=np.array(Ytrain_tensor,dtype='float32')
    
    results_np=results_np[:,0]
    Ytrain_tensor_np=Ytrain_tensor_np[:,0]
    #print("results_np.shape",results_np.shape)
    #print("Ytrain_tensor_np",Ytrain_tensor_np.shape)
    Corr = np.corrcoef(results_np, Ytrain_tensor_np)
    Corr = Corr[1][0]
    R2   = r2_score(results_np, Ytrain_tensor_np)
    # calculate derivative by stepping backward
    MSE.backward()

    # Updating parameters
    optimizer.step()

    # store loss
    MSE_list.append(MSE)

    # print loss
    if(iteration % 200 == 0):
        print('epoch= {}, MSE= {}, RMSE= {}, Corr= {},R2= {})'.format(iteration, MSE,RMSE,Corr,R2))#Corr.data,R2.data))

plt.plot(range(iteration_number),MSE_list)
plt.xlabel("Number of Iterations")
plt.ylabel("MSE")
plt


# In[92]:


predicted_test = cnn(Xtest_tensor).data.numpy()

predicted_test=predicted_test[:,0]
Ytest=Ytest[:,0]
RMSE = sqrt(mean_squared_error(Ytest, predicted_test))
Corr = np.corrcoef(Ytest, predicted_test)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted_test)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted_test)
plt.title('Convolutional neural network. Torch.\n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # Another network builded by varainces of brown channel

# In[82]:


# another network

net = torch.nn.Sequential(
        torch.nn.Linear(1, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


# In[85]:



# Optimization (find parameters that minimize error)
learning_rate = 0.02
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
#a plot for the loss function on our model
MSE_list = []
iteration_number = 1000
for iteration in range(iteration_number):

    # perform optimization with zero gradient
    optimizer.zero_grad()

    results = net(Xtrain_tensor)
    MSE = mse(results, Ytrain_tensor)
    RMSE = sqrt(mse(results, Ytrain_tensor))
    
    
    results_np=results.detach().numpy()
    Ytrain_tensor_np=np.array(Ytrain_tensor,dtype='float32')
    
    results_np=results_np[:,0]
    Ytrain_tensor_np=Ytrain_tensor_np[:,0]
    #print("results_np.shape",results_np.shape)
    #print("Ytrain_tensor_np",Ytrain_tensor_np.shape)
    Corr = np.corrcoef(results_np, Ytrain_tensor_np)
    Corr = Corr[1][0]
    R2   = r2_score(results_np, Ytrain_tensor_np)
    # calculate derivative by stepping backward
    MSE.backward()

    # Updating parameters
    optimizer.step()

    # store loss
    MSE_list.append(MSE)

    # print loss
    if(iteration % 200 == 0):
        print('epoch= {}, MSE= {}, RMSE= {}, Corr= {},R2= {})'.format(iteration, MSE,RMSE,Corr,R2))#Corr.data,R2.data))

plt.plot(range(iteration_number),MSE_list)
plt.xlabel("Number of Iterations")
plt.ylabel("MSE")
plt


# In[87]:


predicted_test = net(Xtest_tensor).data.numpy()

predicted_test=predicted_test[:,0]
Ytest=Ytest[:,0]
RMSE = sqrt(mean_squared_error(Ytest, predicted_test))
Corr = np.corrcoef(Ytest, predicted_test)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted_test)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted_test)
plt.title('Convolutional neural network. Torch.\n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # Another network builded by varainces of brown channel

# In[5]:


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
torch.manual_seed(1)


# In[45]:


plt.scatter(Xtrain_tensor.data.numpy(), Ytrain_tensor.data.numpy())
plt.show()


# In[10]:


class CNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(CNN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


# In[11]:


#sgd optimizer
cnn = CNN(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(cnn)  # net architecture
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
plt.ion() 


# In[13]:


for t in range(100):
    prediction = cnn(Xtrain_tensor)     # input x and predict based on x

    loss = loss_func(prediction, Ytrain_tensor)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(Xtrain_tensor.data.numpy(), Ytrain_tensor.data.numpy())
        plt.plot(Xtrain_tensor.data.numpy(), prediction.data.numpy(), 'b-', lw=5)
        plt.text(0.5, 0, 'MSE=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color':  'black'})
        plt.show()
        plt.pause(0.1)

plt.ioff()


# In[75]:


predicted_test = net(Xtest_tensor)

predicted_test=predicted_test[:,0]

loss = loss_func(predicted_test, Ytest_tensor)
RMSE = sqrt(loss)

predicted_test=predicted_test.detach().numpy()
#print(predicted_test.shape) 
#print(Ytest.shape)
Corr = np.corrcoef(Ytest, predicted_test)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted_test)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted_test)
plt.title('Convolutional network. Torch. (Optimizer SGD, hidden layers=10) \n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# # Another network builded by varainces of brown channel

# In[15]:


#Adam optimizer
class CNN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(CNN, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
cnn = CNN(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(cnn)  # net architecture
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss  

for t in range(100):
    prediction = cnn(Xtrain_tensor)     # input x and predict based on x

    loss = loss_func(prediction, Ytrain_tensor)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 10 == 0:
        
        plt.cla()
        plt.scatter(Xtrain_tensor.data.numpy(), Ytrain_tensor.data.numpy())
        plt.plot(Xtrain_tensor.data.numpy(), prediction.data.numpy(), 'b-', lw=5)
        plt.text(0.5, 0, 'MSE=%.4f' % loss.data.numpy(), fontdict={'size': 15, 'color':  'red'})
        plt.show()
        plt.pause(0.1)

plt.ioff()


# In[72]:


predicted_test = cnn(Xtest_tensor)

predicted_test=predicted_test[:,0]

loss = loss_func(predicted_test, Ytest_tensor)
RMSE = sqrt(loss)

predicted_test=predicted_test.detach().numpy()
#print(predicted_test.shape) 
#print(Ytest.shape)
Corr = np.corrcoef(Ytest, predicted_test)
Corr = Corr[1][0]
R2   = r2_score(Ytest, predicted_test)

#print(predicted.shape)
#print(Ytest.shape)
plt.scatter(Ytest, predicted_test)
plt.title('Convolutional network. Torch.(Optimizer Adam, hidden layers=10)\n RMSE= {}\n Corr.= {}\n R2= {} '.format(RMSE,Corr,R2))
plt.xlabel('True Values')
plt.ylabel('Predicted values')


# In[76]:


from IPython.display import HTML, display
import tabulate
table = [["" ,"CNN. Torch. Brown var.(Optimizer SGD, hidden layers=5)","CNN. Torch. Brown var.(Optimizer SGD, hidden layers=10)","CNN. Torch. Brown var.(Optimizer SGD, hidden layers=20)","CNN. Torch.Brown var.(Optimizer Adam, hidden layers=5)","CNN. Torch. Brown var.(Optimizer Adam, hidden layers=10)","CNN. Torch. Brown var.(Optimizer Adam, hidden layers=20)","CNN.Keras. Brown all (PCA=1200). (Kernel='Normal', activ='Relu')","CNN. Torch.Brown all (PCA=1200)(Optimizer SGD, hidden layers=20)"],
        ['RMSE',5.203,5.41,7.151,7.185,7.152,7.165,3.89,4.27],
        ['Correlation',0.44,0.71,0.749,0.747,0.747,0.747,0.65,0.58],
        ['R2',0.08,0.37,0.261,0.42,0.434,0.4315,0.42,0.303],
        ]
display(HTML(tabulate.tabulate(table, tablefmt='html')))


# Answer:
# 
#     According to results one can conclude that Convolutional network Keras (Brown all (PCA=1200). (Kernel='Normal', activ='Relu') shows best results, where RMSE=3.89, correlation=0.65 and R2=0.42.

# In[ ]:




