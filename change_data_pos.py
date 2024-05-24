import pandas as pd
import matplotlib.pyplot as plt

Y = pd.read_csv("Y_engine_4.csv")


def replace_values(column):
    indices = column[column == 1].index
    if not indices.empty:
        column.loc[indices] = 2
    return column


Y = Y.apply(replace_values)
Y_new = Y[100000::]
print(len(Y))
print(len(Y_new))


plt.figure(figsize=(10, 6))
plt.plot(Y, color='blue')
plt.title("Реальные")
plt.xlabel("Индекс")
plt.ylabel('Y')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Y_new, color='red')
plt.title("Сдвиг на 100 000")
plt.xlabel("Индекс")
plt.ylabel('Y')

plt.show()


