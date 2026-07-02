import gdown
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import drive
drive.mount('/content/drive')


file_id = '1FrbOsgLEzGrpwDodU2iOgmBEKM-gYF4J'
output_file = 'DatasetDS.zip'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output_file, quiet=False)
!unzip - q "/content/DatasetDS.zip"

TrainDIR = "/content/DatasetDS/train"
ValidationDIR = "/content/DatasetDS/validation"
TestDIR = "/content/DatasetDS/test"
CATEGORIES = ["DS", "NoDS"]

SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

tf.keras.utils.set_random_seed(SEED)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TrainDIR,
    labels="inferred",
    label_mode="binary",
    class_names=CATEGORIES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    ValidationDIR,
    labels="inferred",
    label_mode="binary",
    class_names=CATEGORIES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TestDIR,
    labels="inferred",
    label_mode="binary",
    class_names=CATEGORIES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

model = Sequential([
    Input(shape=(*IMG_SIZE, 3)),
    Rescaling(1.0 / 255),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print(f"Algoritmo de optimización: {optimizer.__class__.__name__}")
print(f"Función de pérdida: {model.loss}")
model.summary()

BOLETA = input("Ingresa los 10 caracteres de tu boleta: ").strip()

if len(BOLETA) != 10 or not BOLETA.isalnum():
    raise ValueError(
        "La boleta debe contener exactamente 10 caracteres alfanuméricos")

model_path = f"/content/{BOLETA}_modelo.keras"
graph_path = f"/content/{BOLETA}_precision.png"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.title("Precisión durante el entrenamiento")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.show()

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"Pérdida de prueba: {test_loss:.4f}")
print(f"Precisión de prueba: {test_accuracy:.4f}")
print(f"Modelo guardado en: {model_path}")
print(f"Gráfica guardada en: {graph_path}")
