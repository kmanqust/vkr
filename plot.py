import pandas as pd
# import matplotlib.pyplot as plt
#
# # pd.set_option('display.max_columns', None)
#
X = pd.read_parquet("C:/Users/Salko/Desktop/учеба тпу/4 курс/НИРС/папка хакатона/X_train.parquet")
# Y = pd.read_csv('Y_engine_4.csv')
#
# # for s in Y.columns.tolist():
# #     if "ЭЛЕКТРОДВИГАТЕЛЬ" in s:
# #         print(s)
#
# # Y_ЭКСГАУСТЕР А/М №4_ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №4
# # Y_ЭКСГАУСТЕР А/М №9_ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №9
# # Y_ЭКСГАУСТЕР А/М №7_ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №7
#
# # print(Y['Y_ЭКСГАУСТЕР А/М №4_ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №4'].values)
#
# # print(len(Y))
#
#
# # Создаем столбчатую диаграмму
# plt.figure(figsize=(10, 6))
# plt.plot(Y)
# plt.title("Столбчатая диаграмма")
# plt.xlabel("Индекс")
# plt.ylabel('Y_ЭКСГАУСТЕР А/М №4_ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №4')
# plt.show()
# #
# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # Y = pd.read_csv("Y_engine_100_4.csv")
# #
# # plt.figure(figsize=(10, 6))
# # plt.plot(Y)
# # plt.title("Столбчатая диаграмма")
# # plt.xlabel("Индекс")
# # plt.ylabel('Y_ЭКСГАУСТЕР А/М №4_ЭЛЕКТРОДВИГАТЕЛЬ ДСПУ-140-84-4 ЭКСГ. №4')
# # plt.show()

print(X.head(4))