import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow import keras
import numpy as np

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255
X_train[0]
X_train.shape

X_train = X_train.reshape(-1,28,28,1)
X_train.shape

X_test = X_test.reshape(-1,28,28,1)
X_test.shape

convolutional_neural_network = models.Sequential([
    layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

convolutional_neural_network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
convolutional_neural_network.fit(X_train, y_train, epochs=10)

convolutional_neural_network.evaluate(X_test, y_test)

y_predicted_by_model = convolutional_neural_network.predict(X_test)
y_predicted_by_model[0]   #getting probability score for each class digits

np.argmax(y_predicted[0])

y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]

