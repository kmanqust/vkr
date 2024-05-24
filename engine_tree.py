# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
#
# X = pd.read_parquet("C:/Users/Salko/Desktop/учеба тпу/4 курс/НИРС/папка хакатона/X_train.parquet")[2560000:2660000]
# Y = pd.read_parquet("C:/Users/Salko/Desktop/учеба тпу/4 курс/НИРС/папка хакатона/y_train.parquet")[2560000:2660000]
#
# Y = Y['Y_ЭКСГАУСТЕР А/М №4_ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №4']
#
# columns_to_drop = [col for col in X.columns if "ЭКСГАУСТЕР 4" not in col]
#
# X.drop(columns=columns_to_drop, inplace=True)
#
# X.to_csv('X_engine_100_4.csv', index=False)
# Y.to_csv('Y_engine_100_4.csv', index=False)
#
# s = 0
# v = 0
# for i in Y.values:
#     if i == 2:
#         s += 1
#     if i == 0:
#         v += 1
#
# print(s)
# print(v)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option('display.max_columns', None)

# Загрузка данных
X = pd.read_csv('X_engine_1m_4.csv')
Y = pd.read_csv('Y_engine_1m_4.csv')

# print(X.head(3))

Y_val = pd.read_csv('Y_engine_100t_4.csv')
X_val = pd.read_csv('X_engine_100t_4.csv')

# print(X_val.head(40))

# Разделение данных
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Создание и обучение модели дерева решений
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, Y_train)

# Предсказание на тестовых данных
Y_pred = classifier.predict(X_test)

# Оценка модели на тестовой выборке
accuracy_test = accuracy_score(Y_test, Y_pred)
class_report_test = classification_report(Y_test, Y_pred)

# Предсказание на валидационной выборке
Y_val_pred = classifier.predict(X_val)

# Оценка модели на валидационной выборке
accuracy_val = accuracy_score(Y_val, Y_val_pred)
class_report_val = classification_report(Y_val, Y_val_pred)

# Вывод метрик для тестовой выборки
print("Тестовая выборка:")
print("Accuracy:", accuracy_test)
print("Classification Report:\n", class_report_test)

# Вывод метрик для валидационной выборки
print("\nВалидационная выборка:")
print("Accuracy:", accuracy_val)
print("Classification Report:\n", class_report_val)

