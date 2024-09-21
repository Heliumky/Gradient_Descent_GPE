import numpy as np



import numpy as np

x = np.array([f"x{i}" for i in range(0, 8)])
y = np.array([f"y{i}" for i in range(0, 8)])

X, Y = np.meshgrid(x, y)

# 使用 np.vectorize 來定義一個可以進行字符串加法的向量化函數
def concatenate_elements(a, b):
    return a + b

vectorized_concat = np.vectorize(concatenate_elements)
Z = vectorized_concat(X, Y).T

print("Z:\n", Z)
print("\nFlattened Z:\n", Z.flatten())


n_sites = 6
N = n_sites//2
Ndx = 2**N
for i in range(2**n_sites):
    inds = format(i, '06b')  # 将整数 i 格式化为四位二进制字符串
    xinds, yinds = inds[:], inds[N:]
    print(inds)