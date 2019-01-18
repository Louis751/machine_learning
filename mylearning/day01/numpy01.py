import random

import numpy as np

t1 = np.array([1,2,3])
print(t1)
print(type(t1)) # 数组类型

t2 = np.array(range(10))
print(t2)
print(type(t2))

t3 = np.arange(2,10,2)
print(t3)
print(t3.dtype)
print("*"*50)

t4 = np.array(range(1,4),dtype="i1")
print(t4)
print(t4.dtype)

t5 = np.array([1,1,0,1,0,1],dtype=bool)
print(t5)
print(t5.dtype)

# 调整数据类型
t6 = t5.astype("int8")
print(t6)
print(t6.dtype)

# numpy中小数
t7 = np.array([random.random() for i in range(6)])
print(t7)
print(t7.dtype)

t8 = np.round(t7,2) # 保留两位小数
print(t8)

t9 = random.random()
print("%.2f"%t9)
