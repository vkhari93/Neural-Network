from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from sklearn.metrics import classification_report


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train/X_train.max()
X_test = X_test/X_test.max()

y_train = to_categorical(y_train, 10)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='rmsprop', metrics=['accuracy'], loss='categorical_crossentropy')

model.fit(X_train, y_train, epochs=15)

predict_data = model.predict_classes(X_test)
print(classification_report(y_test, predict_data))

model.save('cifar10.h5')




