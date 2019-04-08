from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from sklearn.metrics import confusion_matrix, classification_report

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# one hot encoding
y_train = to_categorical(y_train, 10)

X_train = X_train/X_train.max()
X_test = X_test/X_test.max()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=2)
predict_data = model.predict_classes(X_test)
print(confusion_matrix(y_test, predict_data))
print(classification_report(y_test, predict_data))

#model.save('mnist.h5')
