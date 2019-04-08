import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix

bank_data = genfromtxt(r'H:\DATA\bank_note_data.txt', delimiter=',')

data_label = bank_data[:, 4]
data_features = bank_data[:, :4]

X_train, X_test, y_train, y_test = train_test_split(data_features, data_label, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model .add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, verbose=2)


y_predict = model.predict_classes(X_test)
matrix = confusion_matrix(y_test, y_predict)
print(matrix)

report = classification_report(y_test, y_predict)
print(report)

model.save('fake_currency.h5')
