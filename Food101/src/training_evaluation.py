import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetV2S, MobileNetV3Large
from tensorflow.keras.mixed_precision import LossScaleOptimizer

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

IMG_SIZE = 224
INITIAL_EPOCHS = 3
NUM_CLASSES = 101


def build_feature_extraction_model(base_model_class, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                   preprocess_layer=None, num_classes=NUM_CLASSES):
    """
    Args:
        base_model_class: A class for the base model (e.g., tf.keras.applications.MobileNet).
        preprocess_layer: Optional preprocessing layer (e.g., tf.keras.layers.Lambda with preprocess function).
        input_shape: The shape of the input data (excluding batch size).
        num_classes: The number of output classes for classification.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    if preprocess_layer:
        x = layers.Lambda(preprocess_layer)(layers.Input(shape=input_shape))
        base_model = base_model_class(input_tensor=x, include_top=False, weights="imagenet")
    else:
        base_model = base_model_class(input_shape=input_shape, include_top=False, weights="imagenet")

    total_params = base_model.count_params()
    
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model, total_params



def plot_loss_curves(history, title, test_name=""):
    """Return separate loss/accuracy curves for training and validation metrics."""
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    plt.figure(figsize=(14,4))
    plt.suptitle(title)

    plt.subplot(121)
    plt.plot(epochs, loss, label="training loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.title("loss")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs, accuracy, label="training accuracy")
    plt.plot(epochs, val_accuracy, label="val accuracy")
    plt.title("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    if len(test_name) > 0:
        plt.savefig(f"{title}_{test_name}.png")

    plt.show()
    

def test_models(models_config, train_data, validation_data, test_data, num_classes=NUM_CLASSES, epochs=INITIAL_EPOCHS):
    """
    Test various models with different configurations and save the results.
    
    Args:
        models_config (list): List of dicts containing model, its name and preprocessor.
        train_data (tf.data.Dataset): Training data.
        val_data (tf.data.Dataset): Validation data.
        num_classes (int): Number of output classes.
        epochs (int): Number of epochs to train the model.
    
    Returns:
        pd.DataFrame: DataFrame with model performance results.
    """
    results = []

    for config in models_config:
        model_class = config['model']
        model_name = model_class.__name__
        preprocess_layer = config['preprocessor']

        checkpoint_callback = callbacks.ModelCheckpoint(f'checkpoints/{model_name}_feature_extraction.keras',
                                                monitor='val_loss',
                                                save_best_only=True,
                                                mode='min',
                                                verbose=0
                                                )

        print(f"Training model: {model_name}...")

        model, total_params = build_feature_extraction_model(model_class, preprocess_layer=preprocess_layer)

        history = model.fit(train_data,
                            validation_data=validation_data,
                            epochs=epochs,
                            callbacks=[checkpoint_callback],
                            verbose=0)

        history_data = {'epoch': history.epoch,
                        'history': history.history}
                        
        with open(f'history/{model_name}_feature_extraction.json', 'w') as f:
            json.dump(history_data, f)
        
        val_accuracy = max(history.history["val_accuracy"])
        test_accuracy = model.evaluate(test_data)[1]
        results.append({"Model": model_name,
                        "Base model params": total_params,
                        "Validation accuracy": val_accuracy,
                        "Test accuracy": test_accuracy})

        plot_loss_curves(history, model_name)

    results = pd.DataFrame(results) 
    results.to_csv("feature_extraction_results.csv", index=False)

    return results.sort_values(by="Test accuracy", ascending=False)


def build_model_for_fine_tuning(base_model_class, input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES, unfrozen_layers=None):
    """
    Args:
        base_model_class: A class for the base model (e.g., tf.keras.applications.MobileNet).
        input_shape: The shape of the input data (excluding batch size).
        num_classes: The number of output classes for classification.
        learning_rate: Learning rate for base optimizer (set to 10x lower than the default)

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    if base_model_class.__name__ == "MobineNetV3Large":
        x = layers.Lambda(applications.mobilenet_v3.preprocess_input)(layers.Input(shape=input_shape))
        base_model = base_model_class(input_tensor=x, include_top=False, weights="imagenet")
    else:    
        base_model = base_model_class(input_shape=input_shape, include_top=False, weights="imagenet")
        
    base_model.trainable = False

    unfrozen_layers = unfrozen_layers if unfrozen_layers else 10
    
    for layer in base_model.layers[-unfrozen_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
        
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    return model


def compare_histories(name, original_history_history_dict, new_history, test_name, initial_epochs=INITIAL_EPOCHS):
    """
    Compares two TensorFlow training histories.
    """
    acc = original_history_history_dict["accuracy"]
    loss = original_history_history_dict["loss"]

    val_acc = original_history_history_dict["val_accuracy"]
    val_loss = original_history_history_dict["val_loss"]

    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(14, 4))
    plt.suptitle(name)
    
    plt.subplot(1,2,1)
    plt.plot(total_acc, label="Training Accuracy")
    plt.plot(total_val_acc, label="Val Accuracy")
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="lower right")
    plt.title("Training and validation accuracy")

    plt.subplot(1,2,2)
    plt.plot(total_loss, label="Training Loss")
    plt.plot(total_val_loss, label="Val Loss")
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc="upper right")
    plt.title("Training and validation loss")

    plt.savefig(f"{name}_{test_name}.png")

    plt.tight_layout()
    plt.show()


def fine_tune_models(test_name, train_data, model_list=[EfficientNetV2S, MobileNetV3Large], add_epochs=22, learning_rate=1e-04,
                     load_previous_weights=False, initial_epoch=0, unfrozen_layers=None):

    results = []

    for model_class in model_list:
    
        name = model_class.__name__
        model = build_model_for_fine_tuning(model_class, unfrozen_layers=unfrozen_layers)

        checkpoint_callback = callbacks.ModelCheckpoint(f'checkpoints/{name}_{test_name}.keras',
                                    monitor='val_loss',
                                    save_best_only=True,
                                    mode='min',
                                    verbose=1
                                    )
        
        if load_previous_weights:
            model.load_weights(f'checkpoints/{name}_feature_extraction.keras')
            with open(f'history/{name}_feature_extraction.json', 'r') as f:
                old_history = json.load(f)

            epochs=INITIAL_EPOCHS + add_epochs
            initial_epoch=old_history["epoch"][-1]
        else:
            epochs=add_epochs

        optimizer = LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))
        
        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        early_stopping = callbacks.EarlyStopping(monitor="val_loss",
                                                 mode="min",
                                                 patience=3,
                                                 restore_best_weights=True)

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                factor=0.2,
                                                patience=2,
                                                verbose=1,
                                                min_lr=1e-7)
    
        print(f"Training model: {name}...")

    
        history = model.fit(train_data,
                            epochs=epochs,
                            initial_epoch=initial_epoch,
                            validation_data=validation_data,
                            callbacks=[checkpoint_callback, early_stopping, reduce_lr],
                            verbose=1)
    
        min_val_loss_index = history.history["val_loss"].index(min(history.history["val_loss"]))
        val_accuracy = history.history["val_accuracy"][min_val_loss_index]
        test_accuracy = model.evaluate(test_data)[1]
        
        results.append({"Model": name,
                        "Validation Accuracy": val_accuracy,
                        "Test Accuracy": test_accuracy})

        if load_previous_weights:
            compare_histories(name, old_history["history"], history, test_name)
        else:
            plot_loss_curves(history, name, test_name)
    
    results = pd.DataFrame(results)
    results.to_csv(f"{test_name}_results.csv", index=False)

    return results.round(3)