import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Импорт RandomForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option('display.max_columns', None)

# Загрузка данных
X = pd.read_csv('X_engine_1m_4.csv')
Y = pd.read_csv('Y_engine_1m_4.csv')

Y_val = pd.read_csv('Y_engine_100t_4.csv')
X_val = pd.read_csv('X_engine_100t_4.csv')

# Разделение данных
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Создание и обучение модели случайного леса
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, Y_train)

# Предсказание на тестовых данных
Y_pred = classifier.predict(X_test)

# Оценка модели на тестовой выборке
accuracy_test = accuracy_score(Y_test, Y_pred)
conf_matrix_test = confusion_matrix(Y_test, Y_pred)
class_report_test = classification_report(Y_test, Y_pred)

# Предсказание на валидационной выборке
Y_val_pred = classifier.predict(X_val)

# Оценка модели на валидационной выборке
accuracy_val = accuracy_score(Y_val, Y_val_pred)
conf_matrix_val = confusion_matrix(Y_val, Y_val_pred)
class_report_val = classification_report(Y_val, Y_val_pred)

# Вывод метрик для тестовой выборки
print("Тестовая выборка:")
print("Accuracy:", accuracy_test)
print("Confusion Matrix:\n", conf_matrix_test)
print("Classification Report:\n", class_report_test)

# Вывод метрик для валидационной выборки
print("\nВалидационная выборка:")
print("Accuracy:", accuracy_val)
print("Confusion Matrix:\n", conf_matrix_val)
print("Classification Report:\n", class_report_val)