import numpy as np
arr = [[1, 5, 7, 33, 39, 52],
        [2, 3, 4, 1, 2, 4],
        [3, 4, 2, 10, 4, 3]]
t = np.array(arr)

# test enumerate
# for i in enumerate(t):
#     print(i)

# test argmax
# print(t)
# print(t.argmax(axis=0))

print(np.where(t==39))