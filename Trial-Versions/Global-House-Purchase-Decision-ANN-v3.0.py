
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers, models, layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tf.keras.backend.clear_session()

import wandb
import optuna

from wandb.integration.keras import WandbMetricsLogger

wandb.require("core")
wandb.login()

ds = pd.read_csv("/global_house_purchase_dataset.csv")
ds.head()

print("NaN values: \n \n ", ds.isna().sum(),
      "\n \n Types: \n \n", ds.dtypes)

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

ds = pd.get_dummies(ds, columns = ["country", "city", "property_type", "furnishing_status"],
                            drop_first = False)
ds.drop("property_id", axis = 1, inplace = True)
ds = ds.astype("float32")

print(ds.head())

print("\n \n Types: \n \n", ds.dtypes)

x = ds.drop("decision", axis=1)
y = ds["decision"] 

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 5)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 1/3, random_state = 5)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# scaler = StandardScaler()
X_train = StandardScaler.fit_transform(X_train)
X_val = StandardScaler.transform(X_val)
X_test = StandardScaler.transform(X_test)

def objective(trial):

    tf.keras.backend.clear_session()

    model = models.Sequential()

    # Optuna suggest first layer's neuron units and activation function
    n_units1 = trial.suggest_int("Layer_1_units", 32, 128)
    activation_1 = trial.suggest_categorical("Activation_1", ["relu", "sigmoid"])
    model.add(layers.Dense(n_units1, activation=activation_1, input_shape=(X_train.shape[1],)))

    # Optuna suggests number of layers
    
    n_layers = trial.suggest_int("n_layers", 1, 10)

    units_per_layer = [n_units1]
    activations_per_layer = [activation_1]
    
    # Optuna suggests activation function, dropout and regularizers
    
    for i in range(n_layers):
        
        n_units = trial.suggest_int(f"Layer_{i+2}_units", 32, 128)
        activation = trial.suggest_categorical(f"Activation_{i+2}", ["relu", "sigmoid"])
        regularizers = trial.suggest_categorical(f"Regularizer_{i+2}", ["L1","L2","L1L2"])
        r_value = trial.suggest_float("regularizer_value", 1e-6, 1e-4, log = True)
        
        
        model.add(layers.Dense(n_units, activation=activation, kernel_regularizer = tf.keras.regularizers.l2(1e-4)))
        
        units_per_layer.append(n_units)
        activations_per_layer.append(activation)

    model.add(layers.Dense(1, activation="sigmoid"))
    
    units_per_layer.append(1)
    activations_per_layer.append("sigmoid")

    # Optuna suggests learning rate and optimizer
    
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    wandb.init(
        project="Global-House-Purchase-Decision-ANN-Trials-3",
        name=f"Trial_{trial.number}",
        reinit=True,
        config={
            "n_layers": n_layers,
            "units_per_layer": units_per_layer,
            "activations_per_layer": activations_per_layer,
            "learning_rate": lr,
            "optimizer": optimizer_name,
        }
    )

    batch_size = trial.suggest_categorical("batch_size", [20, 40, 50, 100])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=batch_size,
        verbose=0, 
        callbacks=[WandbMetricsLogger(log_freq=5)]
    )

    val_loss = min(history.history["val_loss"])
    train_loss = min(history.history["loss"])

    # Penalize overfitting
    score = val_loss + 0.1 * (train_loss - val_loss)

    wandb.finish()

    return score

study = optuna.create_study(study_name = "Proyecto", direction = "minimize")
study.optimize(objective, n_trials = 100)

print("Número de pruebas terminadas: ", len(study.trials))

trial = study.best_trial

print("Mejor intento: ", trial)

print("Valor: ", trial.value)
print("Hiperparámetros: ", trial.params)

from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank

# len(study.trials)

import plotly.io as pio

# pio.renderers.default = "notebook_connected"

plot_parallel_coordinate(study)
plot_param_importances(study)
