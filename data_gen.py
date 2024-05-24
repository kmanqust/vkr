import pandas as pd

X = pd.read_csv("X_engine_100t_4.csv")

print(X.columns.tolist())
# def replace_values(column):
#     indices = column[column == 1].index
#     if not indices.empty:
#         column.loc[indices] = 2
#     return column
#
#
# Y = Y.apply(replace_values)
#
# Y = Y[2100000:3100000]
#
# X = pd.read_csv("X_engine_4.csv")[2000000:3000000]
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
# Y.to_csv('Y_engine_1m_4.csv', index=False)
# X.to_csv('X_engine_1m_4.csv', index=False)
