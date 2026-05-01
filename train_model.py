# FarmCV Sell - Crop Disease Detection using EfficientNetV2
# Author: Mohit Choudhari

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision


# -----------------------------
# Plotting & metrics
# -----------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

"""# Reproducibility"""

SEED = 42
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

"""# Paths"""

DATASET_DIR = "dataset/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
TEST_DIR = "dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

"""# Parameters"""

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

"""# Dataset Loading"""

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.1,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.1,
    subset="validation",
    shuffle=False,
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

"""# Class labels verfication"""

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Number of classes: {NUM_CLASSES}\n{class_names}")

"""# Performance Optimization"""

# train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
train_ds = train_ds.shuffle(125).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

"""# Data Augmentation"""

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.1),
])

"""# Model Architecture"""

base_model = keras.applications.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

"""# Optimizer & Scheduler"""

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=EPOCHS * len(train_ds)
)

optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4
)
# optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

"""# Model Summary"""

model.summary()

"""# Callbacks"""

es_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

callbacks = [es_callback]

"""# Training"""

history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
)

"""# Plot Training Curves"""

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

"""# Evaluate on Test Set"""

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

"""# Confusion Matrix & Per-class Metrics"""

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(16,14))
sns.heatmap(
    cm,
    xticklabels=class_names,
    yticklabels=class_names,
    annot=False,
    cmap="Blues"
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

"""# Save Model"""

model.save("plant_disease_efficientnetv2b0.keras")
model.save_weights("plant_disease_efficientnetv2b0.weights.h5")
print("Model saved successfully.")

# Save Frontend Required Details
import json

# 1. Save class names as JSON
class_names_dict = {
    "class_names": class_names,
    "num_classes": NUM_CLASSES,
    "class_to_index": {name: idx for idx, name in enumerate(class_names)},
    "index_to_class": {idx: name for idx, name in enumerate(class_names)}
}

with open("class_names.json", "w") as f:
    json.dump(class_names_dict, f, indent=2)

# 2. Save model metadata
model_metadata = {
    "model_name": "EfficientNetV2B0",
    "input_shape": [IMG_SIZE, IMG_SIZE, 3],
    "image_size": IMG_SIZE,
    "num_classes": NUM_CLASSES,
    "batch_size": BATCH_SIZE,
    "preprocessing": "efficientnet_v2",
    "model_file": "plant_disease_efficientnetv2b0.keras",
    "weights_file": "plant_disease_efficientnetv2b0.weights.h5",
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss)
}

with open("model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

# 3. Save training history
history_dict = {
    "loss": [float(x) for x in history.history["loss"]],
    "val_loss": [float(x) for x in history.history["val_loss"]],
    "accuracy": [float(x) for x in history.history["accuracy"]],
    "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    "epochs": len(history.history["loss"])
}

with open("training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

# 4. Save preprocessing configuration
preprocessing_config = {
    "image_size": IMG_SIZE,
    "preprocessing_function": "keras.applications.efficientnet_v2.preprocess_input",
    "normalization": "EfficientNetV2 preprocessing (scale to [-1, 1])",
    "input_format": "RGB",
    "data_augmentation": {
        "random_flip": "horizontal",
        "random_rotation": 0.05,
        "random_zoom": 0.05,
        "random_contrast": 0.1
    }
}

with open("preprocessing_config.json", "w") as f:
    json.dump(preprocessing_config, f, indent=2)

# 5. Save complete configuration for frontend
frontend_config = {
    "model": model_metadata,
    "classes": class_names_dict,
    "preprocessing": preprocessing_config,
    "training": history_dict
}

with open("frontend_config.json", "w") as f:
    json.dump(frontend_config, f, indent=2)

print("Frontend details saved successfully:")
print("  - class_names.json")
print("  - model_metadata.json")
print("  - training_history.json")
print("  - preprocessing_config.json")
print("  - frontend_config.json (all-in-one)")

