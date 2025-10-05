#!/usr/bin/env python
"""
Contains the model architecture for cnn_model.pt and cnn_model_online.pt. The files 
cnn_model.pt and cnn_model_online.pt contain the model weights. The weight files are 
used to load the weights into the model architecture. 
"""

import os
os.environ["KERAS_BACKEND"] = "torch"

try:
    import keras_core as keras
except ImportError:
    import keras

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='valid')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='valid')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)

        # Second convolutional block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='valid')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)

        # Flattening and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=6400, out_features=512)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # Convolutional Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Convolutional block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flattening and fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def keras_cnn_model_from_pickle(pickle_data, keras):
    """Build a Keras model from pickle data format using Functional API."""
    try:
        # Use Functional API which is more reliable for pre-loaded weights
        inputs = keras.layers.Input(shape=(50, 50, 1), name='input_layer')

        # Conv Block 1
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv2d_20')(inputs)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid', name='conv2d_21')(x)
        x = keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_10')(x)
        x = keras.layers.Dropout(0.25, name='dropout_15')(x)

        # Conv Block 2
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_22')(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv2d_23')(x)
        x = keras.layers.MaxPooling2D((2, 2), name='max_pooling2d_11')(x)
        x = keras.layers.Dropout(0.25, name='dropout_16')(x)

        # Dense Block
        x = keras.layers.Flatten(name='flatten_5')(x)
        x = keras.layers.Dense(512, activation='relu', name='dense_15')(x)
        x = keras.layers.Dropout(0.5, name='dropout_17')(x)
        outputs = keras.layers.Dense(2, activation='softmax', name='dense_16')(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_model')

        # Build the model to initialize weights
        model.build(input_shape=(None, 50, 50, 1))

        # Set weights from pickle data
        if 'weights' in pickle_data:
            weights = pickle_data['weights']
            if len(weights) == 12:  # 6 layers × 2 weights each
                # Get only trainable layers (skip dropout, pooling, flatten)
                trainable_layers = [layer for layer in model.layers if len(layer.weights) > 0]

                if len(trainable_layers) == 6:  # Should be 4 conv + 2 dense
                    weight_idx = 0
                    for layer in trainable_layers:
                        if len(layer.weights) == 2:  # kernel and bias
                            kernel_weight = weights[weight_idx]
                            bias_weight = weights[weight_idx + 1]
                            layer.set_weights([kernel_weight, bias_weight])
                            weight_idx += 2
                else:
                    # Fallback: set all weights at once
                    model.set_weights(weights)
            else:
                raise ValueError(f"Expected 12 weight arrays, got {len(weights)}")
        else:
            raise ValueError("No weights found in pickle data")

        return model

    except Exception as e:
        raise ValueError(f"Failed to build Keras model from pickle: {e}")
        
