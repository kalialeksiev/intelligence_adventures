import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the data so that it is 4 dimensional, i.e.
# num_examples x height x width x num_channels
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# reshape the labels so that they each label is
# a vector of length 10
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# create the models
model = keras.models.Sequential()
# add 2 convolutional layers (immediately followed by max pooling)
model.add(Conv2D(filters=32, kernel_size=[5, 5], strides=[1, 1],
                 padding='same', activation='tanh', use_bias=True,
                 input_shape=[28, 28, 1]))
model.add(MaxPool2D(pool_size=[2, 2], strides=[1, 1], padding='same'))
model.add(Conv2D(filters=64, kernel_size=[5, 5], strides=[1, 1],
                 padding='same', activation='tanh', use_bias=True))
model.add(MaxPool2D(pool_size=[2, 2], strides=[1, 1], padding='same'))
# flatten the output from the convolutional layers (which should have shape
# num_training_examples x 7 x 7 x 64)
model.add(Flatten())
# add 2 dense layers
model.add(Dense(units=1000, activation='tanh', use_bias=True))
model.add(Dense(units=10, activation='softmax', use_bias=True))

model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=X_train, y=y_train, batch_size=100,
          epochs=2, validation_split=0.1)

scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





