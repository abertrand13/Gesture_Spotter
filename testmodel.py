import tensorflow as tf
import keras
import numpy as np
import sys
import os

class residualModule(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(residualModule,self).__init__()
        self.n_filters = n_filters
        self.merge_input = tf.keras.layers.Conv2D(
                self.n_filters, (1, 1), padding='same',
                activation='relu',
                kernel_initializer='he_normal')
        self.conv1 = tf.keras.layers.Conv2D(
            self.n_filters, (3, 3), padding='same',
            activation='relu',
            kernel_initializer='he_normal')
        self.conv2 = tf.keras.layers.Conv2D(
            self.n_filters, (3, 3), padding='same',
            activation='linear',
            kernel_initializer='he_normal')
        self.BatchNormalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        self.add = tf.keras.layers.Add()

    def __call__(self, layer_in):
        merge_input = layer_in
        # check if the number of filters needs to be increased
        if layer_in.shape[-1] != self.n_filters:
            merge_input = self.merge_input(layer_in)

        conv1 = self.conv1(layer_in)
        conv2 = self.conv2(conv1)

        # add filters
        layer_out = self.add([conv2, merge_input])
        layer_out = self.BatchNormalization(layer_out)
        layer_out = self.activation(layer_out)

        return layer_out

def resNet_LSTM(input_shape1,
                input_shape2,
                  output_shape,
                  rnn='LSTM',
                  multi_rnn=True,
                  attention=True,
                  dropout=0.2):
    """
    Neural network with residual blocks
    Accuracy = 0.96
    """

    input_shape = (input_shape1, input_shape2)

    input_layer = tf.keras.Input(shape=input_shape)

    reshape_layer = tf.keras.layers.Reshape(input_shape=input_shape, target_shape=(input_shape1, input_shape2, 1))(input_layer)

    x = tf.keras.layers.Conv2D(64, (6, 4), padding='same', strides=(2, 2), activation='relu')(reshape_layer)
    x = tf.keras.layers.MaxPool2D((3, 2), padding='same', strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = residualModule(64)(x)
    # x = residualModule(128)(x)
    x = tf.keras.layers.MaxPool2D((3, 2))(x)

    x = residualModule(128)(x)
    # x = residualModule(256)(x)

    layer_out = tf.keras.layers.Conv2D(1, kernel_size=(3, 3),
                                       padding='same', activation='relu')(x)
    layer_out = tf.keras.layers.BatchNormalization()(layer_out)
    layer_out = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.squeeze(x, -1), name='squeeze_dim')(layer_out)
    # shape = keras.ops.shape(x)
    # print(shape)
    # layer_out = tf.keras.layers.Reshape([shape[1], shape[2]*shape[3]])(x)

    x = tf.keras.layers.Dense(512, activation='relu')(layer_out)

    # LSTM Layer
    if rnn not in ['LSTM', 'GRU']:
        raise ValueError(
            'rnn should be equal to LSTM or GRU. No model generated...')

    if rnn == 'LSTM':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            128, return_sequences=True, dropout=dropout))(x)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                128, return_sequences=True, dropout=dropout))(layer_out)

    # GRU Layer
    if rnn == 'GRU':
        layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            400, return_sequences=True, dropout=dropout))(x)
        if multi_rnn:
            layer_out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                400, return_sequences=True, dropout=dropout))(layer_out)

    # Attention Layer
    if attention:
        query, value = tf.keras.layers.Lambda(
            lambda x: tf.split(x, num_or_size_splits=2, axis=2))(layer_out)
        layer_out = tf.keras.layers.Attention(name='Attention')([query, value])

    # Classification Layer
    outputs = tf.keras.layers.Flatten()(layer_out)
    # outputs = tf.keras.layers.Dense(512, activation='relu')(outputs)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(outputs)

    model = tf.keras.Model(inputs=[input_layer], outputs=[outputs])
    return model

input_shape1 = 66 # (22 joints * [x,y,z] for each joint)
input_shape2 = 30 # time window
output_shape = 15 # the number of different gestures we want to detect
learning_rate = .0001
epochs = 100
# rnn = 'LSTM'
multiRNN = False
# attention = True
# dropout = 0.2


model = resNet_LSTM(input_shape1, input_shape2, output_shape, multi_rnn=multiRNN)
model.compile(loss='sparse_categorical_crossentropy',
				optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
				metrics=["sparse_categorical_crossentropy", "accuracy"])

model.summary()

datasetFolder = sys.argv[1] if (len(sys.argv) > 1) else  "." # default to current directory

# load data
# train_data = np.load("train_data.npy")
# train_labels = np.load("train_labels.npy")
train_data = np.load(os.path.join(datasetFolder, "train_data.npy"))
train_labels = np.load(os.path.join(datasetFolder, "train_labels.npy"))

# test_data = np.load("test_data.npy")
# test_labels = np.load("test_labels.npy")
test_data = np.load(os.path.join(datasetFolder, "test_data.npy"))
test_labels = np.load(os.path.join(datasetFolder, "test_labels.npy"))

# print(train_data)
print(train_data.shape)
# print(train_labels)

# things for saving
checkpoint_path = sys.argv[1] + "/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, 
													save_weights_only=True,
													verbose=1)

# training the model
# model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels), callbacks=[cp_callback])

model.load_weights("DatasetParse_v4/cp-0097.weights.h5")
model.evaluate(test_data, test_labels, verbose=2)

model.export("DatasetParse_v4/saved_model") # SavedModel format for TF Serving compat
