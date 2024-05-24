import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Загрузка и предобработка данных
X = pd.read_csv('X_engine_1m_4.csv')
Y = pd.read_csv('Y_engine_1m_4.csv') # Укажите здесь правильное имя столбца с метками

# Преобразование меток в категориальный формат
Y = to_categorical(Y)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)

# Создание модели
model = Sequential()
model.add(Dense(96, input_dim=X_train.shape[1], activation='relu'))  # Первый слой
model.add(Dense(64, activation='relu'))  # Второй слой
model.add(Dense(Y_train.shape[1], activation='softmax'))  # Выходной слой

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

# Оценка модели
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Дополнительно можно предсказать и оценить модель на валидационной выборке
X_val = pd.read_csv('X_engine_100t_4.csv')
Y_val = pd.read_csv('Y_engine_100t_4.csv')
Y_val = to_categorical(Y_val)
X_val_scaled = scaler.transform(X_val)
loss_val, accuracy_val = model.evaluate(X_val_scaled, Y_val)
print(f"Validation Accuracy: {accuracy_val*100:.2f}%")