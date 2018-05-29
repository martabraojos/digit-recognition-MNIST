
# coding: utf-8

# ## DIGIT RECOGNITION CHALLENGE

# In[1]:


#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn as skl

import matplotlib.pyplot as plt


# In[2]:


#import datasets. It is already separated in test and train. I am going to use only the "train" set, and then divide it, 
#as I want to be able to know how good is the performance of my model ("test" does not include labels)
dataset = pd.read_csv('C:/Users/Marta/Desktop/Digit Recognition mnist Challenge/train.csv')


# In[3]:


#I want to see the size
print(dataset.shape)


# In[4]:


#look at the datasets
print(type(dataset))
print(dataset.head(5))
print(dataset.shape)
print(dataset.dtypes)
print(dataset.describe())
#we realize it is a set of pixels with a number that indicates if it is coloured and how intensely
#The numbers that indicates coloured is an integer


# In[5]:


#Many pixels has never been coloured (is expected, noone draw a number in the vey corner of the space e.g.)
print(dataset['pixel5'].values.sum())
print(dataset['pixel500'].values.sum())
dataset.sum()


# In[6]:


#Lets check if there are missing values, null, na or nan. There are not! 
dataset.isnull().sum().sum()


# In[7]:


#Lets divide now our dataset into a train set and a test set, 80-20%
np.random.seed(7) #I want to obtain the same results when I rerun this, so I put a seed 
indicator = np.random.rand(len(dataset),) < 0.8


# In[8]:


#I divide the dataset
train = dataset[indicator]
test = dataset[~indicator]


# In[9]:


#I want to see if the distribution of actual numbers is even 
labelsarray = dataset.iloc[:,0]
labelsarray.hist()
plt.show()


# In[10]:


#A fun way to see the numbers!!

a = dataset.loc[40].values[1:]  #select an instance from the dataset, without label
type(a) #it is an array
a.shape #because of the size, 784=2*2*2*2*7*7, I imagine that they are painting in a square 28*28
b = a.reshape(28,28)  #we put them in a matrix (but when printing this, it cuts and looks weird...)
#print(b)
for i in range(28):  #little loop to keep the shape! you could also print *  instead of numbers
    for j in range(28):
        if j==0:
            print("\n",end="")
        if b[i,j] == 0:
            print("   ",end="")
        else:
            print(b[i,j],end=" ")
            
            #note that the numbers in the border tends to be smaller than the ones in the center, because of the pen preasure


# In[11]:


#I will prepare my dataframe for the classifier algorithm; I separate between features and labels
print(train.shape)
features_train = train.iloc[:,1:]
print(features_train.shape)
labels_train = train.iloc[:,0]
print(labels_train.shape)

features_test = test.iloc[:,1:]
labels_test = test.iloc[:,0]


# ## First approach; Decision Tree

# In[12]:


#I import my method 
from sklearn.tree import DecisionTreeClassifier


# In[13]:


#lets define a decision tree algorithm for this case   (THE PARAMETERS I AM USING, I CHOOSED THEM BY A RULE OF THUMB, TRYING)
mytree = DecisionTreeClassifier(criterion = 'entropy',random_state = 7, max_depth=12, min_samples_leaf=9)


# In[14]:


#Train the model!
mytree.fit(features_train,labels_train)


# In[15]:


#Lets try to predict with it
predictions = mytree.predict(features_test)


# In[16]:


#just a first look...
labels_test.values


# In[17]:


#Accuracy within the training set AS THE LABELS ARE EVENLY DISTRIBUTED, THIS IS A GOOD MEASURE OF HOW GOOD IS MY MODEL
actrain = sum(mytree.predict(features_train)==labels_train.values)/len(mytree.predict(features_train))
#Accuracy
actest = sum(predictions==labels_test.values)/len(predictions)
print(actrain)
print(actest)


# ##### I am going to change the dataset; I do not want to see a scale of intensity anymore, I will change it to 0 and 1

# In[18]:


#preparing dataset again!
newdataset = dataset[:]
newdataset[newdataset > 200] = 1 #This avoid little noise around image and also avoid changes in labels


# In[19]:


#Dividing dataset again
np.random.seed(7) #I want to obtain the same results when I rerun this, so I put a seed 
newindicator = np.random.rand(len(dataset),) < 0.8


# In[20]:


#I divide the dataset
newtrain = newdataset[newindicator]
newtest = newdataset[~newindicator]


# In[21]:


#preparing dataset again!
features_newtrain = newtrain.iloc[:,1:]
labels_newtrain = newtrain.iloc[:,0]

features_newtest = newtest.iloc[:,1:]
labels_newtest = newtest.iloc[:,0]


# In[22]:


#Train the new model!
mytree.fit(features_newtrain,labels_newtrain)


# In[23]:


#Lets try to predict with it
newpredictions = mytree.predict(features_newtest)


# In[24]:


#Accuracy within the new training set AS THE LABELS ARE EVENLY DISTRIBUTED, THIS IS A GOOD MEASURE OF HOW GOOD IS MY MODEL
newactrain = sum(mytree.predict(features_newtrain)==labels_newtrain.values)/len(mytree.predict(features_newtrain))
#Accuracy
newactest = sum(newpredictions==labels_newtest.values)/len(newpredictions)
print(newactrain)
print(newactest)


# ##### Same accuracy...

# ## Just for fun, I would like to see if doing a k-mean method it cluster the digits

# In[25]:


#I am going to use just the new dataset (without the labels), as I think it will perform better that way
features_newdataset = newdataset.iloc[:,1:]


# In[26]:


#I import my method 
from sklearn.cluster import KMeans 


# In[27]:


#I adjust the method
kmeans = KMeans(n_clusters=10)  


# In[28]:


#training!
kmeans.fit(features_newdataset)


# In[29]:


kmeans.labels_


# In[30]:


labels_newdataset = newdataset.iloc[:,0]


# In[31]:


for i in range(10):
    labels_newdataset[kmeans.labels_ == i].hist()
    plt.show()


#  not at all what we expected. This is not a good method for this problem.
