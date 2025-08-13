import tensorflow as tf
import numpy as np
from tensorflow import keras

EPOCHS = 20
BATCH_SIZE = 256
NB_CLASSES = 10
N_HIDDEN = 2048
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3
VERBOSE = 1

path = r'C:\Users\user\Desktop\ML\November 12\Data\mnist.npz'

with np.load(path) as data:
    X_train, Y_train = data['x_train'], data['y_train']
    X_test, Y_test = data['x_test'], data['y_test']

RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED).astype('float32') / 255.0
X_test = X_test.reshape(10000, RESHAPED).astype('float32') / 255.0
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED, ), activation='relu', name='Input_Layer'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN, activation='relu', name='Dense_Layer'))
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(NB_CLASSES, activation='softmax', name='Output_Layer'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

predictions = model.predict(X_test)