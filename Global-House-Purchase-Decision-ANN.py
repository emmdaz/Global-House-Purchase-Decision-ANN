#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import wandb
import optuna

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.require("core")
wandb.login()


# In[3]:


# Carga de los datos

ds = pd.read_csv("/global_house_purchase_dataset.csv")
ds.head()


# In[4]:


print("NaN values: \n \n ", ds.isna().sum(),
      "\n \n Types: \n \n", ds.dtypes)


# In[5]:


fig, axes = plt.subplots(2, 2, figsize = (10,10))
axes = axes.flatten()

sns.countplot(ds, x = "country", color = "purple", ax = axes[0])
axes[0].tick_params(axis="x", rotation=90)
axes[0].set_title("country")

sns.countplot(ds, x = "city", color = "grey", ax = axes[1])
axes[1].tick_params(rotation = 90)
axes[1].set_title("city")

sns.countplot(ds, x = "property_type", color = "orange", ax = axes[2])
axes[2].tick_params(rotation = 90)
axes[2].set_title("property_type")

sns.countplot(ds, x = "furnishing_status", color = "green", ax = axes[3])
axes[3].tick_params(rotation = 90)
axes[3].set_title("furnishing_status")

plt.tight_layout()
plt.show() 


# In[6]:


# One hot encoding
ds = pd.get_dummies(ds, columns = ["country", "city", "property_type", "furnishing_status"],
                            drop_first = True)
ds.drop("property_id", axis = 1, inplace = True)
ds = ds.astype("float32")

ds.head()


# In[7]:


print("\n \n Types: \n \n", ds.dtypes)


# In[8]:


x = ds.iloc[:, :-1].values
y = ds.iloc[:, -1].values

print("These are the x values: \n", x,"\n \n These are the y values: \n", y)


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 5)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 1/3, random_state = 5)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))


# In[ ]:




