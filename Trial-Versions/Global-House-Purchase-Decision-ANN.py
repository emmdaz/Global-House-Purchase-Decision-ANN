#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers, models, layers

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


tf.keras.backend.clear_session()


# In[3]:


import wandb
import optuna

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbCallback

wandb.require("core")
wandb.login()


# In[4]:


# Carga de los datos

ds = pd.read_csv("/global_house_purchase_dataset.csv")
ds.head()


# In[5]:


print("NaN values: \n \n ", ds.isna().sum(),
      "\n \n Types: \n \n", ds.dtypes)


# In[6]:


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


# In[7]:


# One hot encoding

ds = pd.get_dummies(ds, columns = ["country", "city", "property_type", "furnishing_status"],
                            drop_first = False)
ds.drop("property_id", axis = 1, inplace = True)
ds = ds.astype("float32")

ds.head()


# In[8]:


print("\n \n Types: \n \n", ds.dtypes)


# In[9]:


x = ds.drop("decision", axis=1)
y = ds["decision"] 


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 5)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 1/3, random_state = 5)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))


# In[11]:


X_train.shape[1]


# In[12]:


def objective(trial):
    
    model = models.Sequential()
    
    # Optuna suggest activation function and neurons for the first layer
    
    n_units1 = trial.suggest_int("Layer_1", 32, 128)
    
    Activation_1_options = ["sigmoid", "relu", "tanh",]
    Activation_1 = trial.suggest_categorical("Activation_1", Activation_1_options)

    if Activation_1 == "sigmoid":
        model.add(layers.Dense(n_units1, activation = "sigmoid", input_shape=(X_train.shape[1],)))

    elif Activation_1 == "relu":
        model.add(layers.Dense(n_units1, activation = "relu", input_shape=(X_train.shape[1],)))

    elif Activation_1 == "tanh":
        model.add(layers.Dense(n_units1, activation = "tanh", input_shape=(X_train.shape[1],)))
    
    # Optuna suggest number of layers
    
    n_layers = trial.suggest_int("n_layers", 10, 20)

    # Optuna suggests number of neurons and activation function per layer
    
    units_per_layer = [n_units1]
    function_per_layer = [Activation_1]
    
    for i in range(n_layers):
        
        n_units = trial.suggest_int(f"Neurons in layer_{i}", 32, 128)
        
        function_options = ["sigmoid", "relu", "tanh",]
        function_selected = trial.suggest_categorical(f"Activation_{i}", function_options)
        
        units_per_layer.append(n_units)
        function_per_layer.append(function_selected)

        if function_selected == "sigmoid":
            model.add(layers.Dense(n_units, activation = "sigmoid"))

        elif function_selected == "relu":
            model.add(layers.Dense(n_units, activation = "relu"))

        elif function_selected == "tanh":
            model.add(layers.Dense(n_units, activation = "tanh"))
    
    model.add(layers.Dense(16, activation = "softmax"))
    model.add(layers.Dense(1, activation = "sigmoid"))
    
    units_per_layer += [16, 1]
    function_per_layer += ["softmax", "sigmoid"]

    # Optuna suggests learning rate

    lr = trial.suggest_float("Learning_rate", 1e-8, 1e-1, log = True)

    # Optuna suggests optimizer
    
    optimizer_options = ["sgd", "adam", "rmsprop"]
    optimizer_selected = trial.suggest_categorical("Optimizer", optimizer_options)

    if optimizer_selected == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    elif optimizer_selected == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate = lr)

    elif optimizer_selected == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr)
        
    # Optuna suggests batch size
    
    batch_size = trial.suggest_categorical("batch_size", [20, 40, 50, 100])
        
    wandb.init(
        project = "Global-House-Purchase-Decision-ANN-Trials",
        name = f"Trial_{trial.number}",
        reinit=True,
        config = {
            "N_layers": n_layers,
            "Units_per_layer": units_per_layer,
            "Activations_per_layer": function_per_layer,
            "Learning_rate": lr,
            "batch_size": batch_size
        }
    )

    # Compilamos el modelo

    model.compile(optimizer = optimizer,
                  loss = "binary_crossentropy",
                  metrics = ["accuracy"])
    
    model.fit(X_train, y_train, epochs = 50, batch_size = batch_size,
              callbacks = [WandbMetricsLogger(log_freq = 5)],
              verbose = 0)
    
    wandb.finish()
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose = 1)

    return accuracy


# In[ ]:


study = optuna.create_study(study_name = "Proyecto", direction = "maximize")
study.optimize(objective, n_trials = 100)


# In[ ]:


print("Número de pruebas terminadas: ", len(study.trials))

trial = study.best_trial

print("Mejor intento: ", trial)

print("Valor: ", trial.value)
print("Hiperparámetros: ", trial.params)


# In[ ]:


from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank


# In[ ]:


len(study.trials)


# In[ ]:


import plotly.io as pio
pio.renderers.default = "notebook_connected"


# In[ ]:


plot_parallel_coordinate(study)


# In[ ]:


plot_param_importances(study)


# In[ ]:




