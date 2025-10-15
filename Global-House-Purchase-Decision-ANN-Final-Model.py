import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers, models, layers
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("TensorFlow is using the GPU \n", gpus)
else:
    print("No GPU detected.")

import wandb 

from wandb.integration.keras import WandbMetricsLogger

wandb.require("core")
wandb.login()

ds = pd.read_csv("/global_house_purchase_dataset.csv")

print("NaN values: \n \n ", ds.isna().sum(),
      "\n \n Types: \n \n", ds.dtypes)

# One hot encoding

ds = pd.get_dummies(ds, columns = ["country", "city", "property_type", "furnishing_status"],
                            drop_first = False)
ds.drop("property_id", axis = 1, inplace = True)
ds = ds.astype("float32")

# Separación de los datos

x = ds.drop("decision", axis=1)
y = ds["decision"] 

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.3, random_state = 5)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 1/3, random_state = 5)

# Estandarización de los datos

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Creación del modelo

wandb.init(
    project="Global-House-Purchase-Decision-ANN-Model-1",
    name = "Entramiento 2",
    config={
        "Layer1": 108,
        "Activation_1": "sigmoid",
        "Dropout1": 0.0,
        "Layer2": 73,
        "Activation_2": "relu",
        "Dropout2": 0.0,
        "Layer3": 34,
        "Actiavtion_3": "sigmoid",
        "Optimizer": "adam",
        "Metric": "accuracy",
        "Epochs": 32,
        "Batch_size": 100,
        "Eta": 0.00014359104323874677,
        "Regularizer": "L1L2",
        "L1": 0.00001508797241665705,
        "L2": 0.00000324211243183006,
        "Loss": "binary_crossentropy"
    }
)

model = models.Sequential()
model.add(layers.Dense(108, activation = "sigmoid",
                       input_shape = (X_train.shape[1],)))
model.add(layers.Dense(73, activation = "relu",
                      kernel_regularizer = tf.keras.regularizers.L1L2(0.00001508797241665705,0.00000324211243183006)))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(optimizer = Adam(0.00014359104323874677), loss = "binary_crossentropy",
             metrics = ["accuracy"])
model.summary()


history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs = 30,
        batch_size = 40,
        verbose = 1, 
        callbacks = [WandbMetricsLogger(log_freq=5)]
    )

wandb.finish()

# Se guarda el modelo

model.save("model_f.h5")

# Evaluación del modelo

model_ev = keras.models.load_model("model_f.h5")

loss, accuracy = model_ev.evaluate(X_val, y_val, verbose=1)

print(f"Test Loss: {loss:}")
print(f"Test Accuracy: {accuracy:}")

import seaborn as sns
from sklearn.metrics import confusion_matrix

y_pred = model_ev.predict(X_val)
y_pred_classes = (y_pred > 0.5).astype("int32")

cm = confusion_matrix(y_val, y_pred_classes)
sns.heatmap(cm, annot = True, fmt = "d", cmap = "rocket")

plt.xlabel("Predicción")
plt.ylabel("Valor real")
plt.show()

plt.plot(history.history["accuracy"], label = "Precisión durante el entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión durante validación (Test)")
plt.title("Precisión del modelo")

plt.grid()
plt.legend()
plt.show()


plt.plot(history.history["loss"], label = "Pérdida durante el entrenamiento")
plt.plot(history.history["val_loss"], label = "Pérdida durante la validación (Test)")
plt.title("Función de costo del modelo")

plt.grid()
plt.legend()
plt.show()
