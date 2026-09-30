import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

DATASET_DIR = "data/processed"
MODEL_PATH = "models/skin_cancer_model.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(dataset_dir=DATASET_DIR, model_path=MODEL_PATH):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    num_classes = len(train_ds.class_names)
    model = build_model(num_classes)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[checkpoint, early_stopping],
    )

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the skin-cancer classifier.")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="Dataset directory for processed images")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to save the trained model")
    args = parser.parse_args()

    main(dataset_dir=args.dataset_dir, model_path=args.model_path)
