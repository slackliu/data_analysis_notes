# 来源于: <https://github.com/rougier/numpy-100>    
https://blog.csdn.net/qq_39362996/article/details/90204737


#### 1. Import the numpy package under the name `np` (★☆☆)
引入Numpy并去名为np

```python
import numpy as np
```

#### 2. Print the numpy version and the configuration (★☆☆)
打印出Numpy的版本跟信息

```python
print(np.__version__)
print(np.__config__)
```

#### 3. Create a null vector of size 10 (★☆☆)
创建一个长度为10的零向量

```python
print(np.zeros(10))
```

#### 4.  How to find the memory size of any array (★☆☆)
获取数组所占内存大小

```python
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
```

#### 5.  How to get the documentation of the numpy add function from the command line? (★☆☆)
怎么用命令行获取numpy add函数的文档说明？

```python
print(np.info(np.sum))
```

#### 6.  Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
创建一个长度为10的零向量，并把第五个值赋值为1

```python
z = np.zeros(10)
z[4] = 1
print(z)
```

#### 7.  Create a vector with values ranging from 10 to 49 (★☆☆)
创建一个值域为10到49的向量

```python
z = np.arange(10, 50)
print(z)
```

#### 8.  Reverse a vector (first element becomes last) (★☆☆)
将一个向量进行反转（第一个元素变为最后一个元素）

```python
z = np.arange(10, 50)
z = z[::-1]
print(z)
```

#### 9.  Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)
创建一个3x3的矩阵，值域为0到8
```python
z = np.arange(9)
z = z.reshape(3, 3)
print(z)
```

#### 10. Find indices of non-zero elements from \[1,2,0,0,4,0\] (★☆☆)
从数组[1, 2, 0, 0, 4, 0]中找出非0元素的位置索引
```python
z = np.nonzero([1, 2, 0, 0, 4, 0])
print(z)
```

#### 11. Create a 3x3 identity matrix (★☆☆)
创建一个3x3的单位矩阵
```python
z = np.eye(3)
print(z)
```

#### 12. Create a 3x3x3 array with random values (★☆☆)
创建一个3x3x3的随机数组
```python
z = np.random.random((3, 3, 3))
print(z)
```

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
创建一个10x10的随机数组，并找出该数组中的最大值与最小值
```python
z = np.random.random((10, 10))
print(z)
print(z.max())
print(z.min())
```

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)
创建一个长度为30的随机向量，并求它的平均值
```python
z = np.random.random(30)
print(z)
print(z.mean())
```

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
创建一个2维数组，该数组边界值为1，内部的值为0 
```python
Z = np.ones((10, 10))
Z[1:-1, 1:-1] = 0
print(Z)
```

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
如何用0来填充一个数组的边界？
```python
Z = np.ones((10, 10))
# Z[0, :] = 0
# Z[-1, :] = 0
# Z[:, 0] = 0
# Z[:, -1] = 0
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
print(Z)
```
```python
Z = np.ones((10, 10))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)
```
#### 17. What is the result of the following expression? (★☆☆)
下面表达式运行的结果是什么？
```python
# 表达式                           # 结果
0 * np.nan                        nan
np.nan == np.nan                  False
np.inf > np.nan                   False
np.nan - np.nan                   nan
0.3 == 3 * 0.1                    False
```

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
创建一个5x5的矩阵，且设置值1, 2, 3, 4在其对角线下面一行
```python
z = np.diag([1,2,3,4],k=-2)
print(z)

```

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
 创建一个8x8的国际象棋棋盘矩阵（黑块为0，白块为1）

```python
Z = np.zeros((8, 8), dtype=int)
Z[1::2, ::2] = 1
# 这种是步长为2的
Z[::2, 1::2] = 1
print (Z)
```

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
思考一下形状为(6, 7, 8)的数组的形状，且第100个元素的索引(x, y, z)分别是什么？

```python
print (np.unravel_index(100, (6, 7, 8)))
```
其过程类似：首先构造一个数组arr2=np.array(range(6*7*8)).reshape((6,7,8));返回indices中的元素值在数组arr2中对应值的索引位置。

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
用tile函数创建一个8x8的棋盘矩阵
'''python
Z = np.tile(np.array([[1, 0], [0, 1]]), (4, 4))
print (Z)
'''


#### 22. Normalize a 5x5 random matrix (★☆☆)
对5x5的随机矩阵进行归一化

```python
Z = np.random.random((5, 5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z-Zmin)/(Zmax-Zmin)
print (Z)
```

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
创建一个dtype来表示颜色(RGBA)

```python
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)])
c = np.array((255, 255, 255, 1), dtype=color)
print(c)

```

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
一个5x3的矩阵和一个3x2的矩阵相乘，结果是什么？

```python
Z = np.dot(np.zeros((5, 3)), np.zeros((3, 2)))
# 或者
# Z = np.zeros((5, 3))@ np.zeros((3, 2))
print (Z)
```

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
给定一个一维数组把它索引从3到8的元素求相反数

```python
Z = np.arange(11)
Z[(3 <= Z) & (Z < 8)] *= -1
print (Z)
```

#### 26. What is the output of the following script? (★☆☆)
下面的脚本的结果是什么？

```python
# Author: Jake VanderPlas               # 结果
 
print(sum(range(5),-1))                 9
from numpy import *                     
print(sum(range(5),-1))                 10    #numpy.sum(a, axis=None)
```

#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
关于整形的向量Z下面哪些表达式正确？

```python
Z**Z                        True
2 << Z >> 2                 False
Z <- Z                      True
1j*Z                        True  #复数           
Z/1/1                       True
Z<Z>Z                       False
```

#### 28. What are the result of the following expressions?
下面表达式的结果分别是什么？

```python
np.array(0) / np.array(0)                           nan
np.array(0) // np.array(0)                          0
np.array([np.nan]).astype(int).astype(float)        -2.14748365e+09
```

#### 29. How to round away from zero a float array ? (★☆☆)
如何从零位开始舍入浮点数组？ 

```python
# Author: Charles R Harris
 
Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))
```

#### 30. How to find common values between two arrays? (★☆☆)
如何找出两个数组公共的元素? 

```python
Z1 = np.random.randint(0, 10, 10)
Z2 = np.random.randint(0, 10, 10)
print (np.intersect1d(Z1, Z2))
```

#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)
如何忽略numpy的警告信息（不推荐）?

```python
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0
 
# Back to sanity
_ = np.seterr(**defaults)
 
# 另一个等价的方式， 使用上下文管理器（context manager）
with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0
```

An equivalent way, with a context manager:

```python

```

#### 32. Is the following expressions true? (★☆☆)
下面的表达式是否为真?

```python
np.sqrt(-1) == np.emath.sqrt(-1)     False
```

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
 如何获得昨天，今天和明天的日期? 

```python
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
```

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
怎么获得所有与2016年7月的所有日期? 

```python
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print (Z)
```

#### 35. How to compute ((A+B)\*(-A/2)) in place (without copy)? (★★☆)


```python
A = np.ones(3) * 1
B = np.ones(3) * 1
C = np.ones(3) * 1
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
np.multiply(A, B, out=A)
```

#### 36. Extract the integer part of a random array using 5 different methods (★★☆)


```python

```

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)


```python

```

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)


```python

```

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)


```python

```

#### 40. Create a random vector of size 10 and sort it (★★☆)


```python

```

#### 41. How to sum a small array faster than np.sum? (★★☆)


```python

```

#### 42. Consider two random array A and B, check if they are equal (★★☆)


```python


```

#### 43. Make an array immutable (read-only) (★★☆)


```python

```

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)


```python

```

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)


```python

```

#### 46. Create a structured array with `x` and `y` coordinates covering the \[0,1\]x\[0,1\] area (★★☆)


```python

```

####  47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))


```python

```

#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)


```python

```

#### 49. How to print all the values of an array? (★★☆)


```python

```

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)


```python

```

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)


```python

```

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)


```python

```

#### 53. How to convert a float (32 bits) array into an integer (32 bits) in place?


```python

```

#### 54. How to read the following file? (★★☆)


```python

```

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)


```python

```

#### 56. Generate a generic 2D Gaussian-like array (★★☆)


```python

```

#### 57. How to randomly place p elements in a 2D array? (★★☆)


```python

```

#### 58. Subtract the mean of each row of a matrix (★★☆)


```python

```

#### 59. How to I sort an array by the nth column? (★★☆)


```python

```

#### 60. How to tell if a given 2D array has null columns? (★★☆)


```python

```

#### 61. Find the nearest value from a given value in an array (★★☆)


```python

```

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)


```python

```

#### 63. Create an array class that has a name attribute (★★☆)


```python

```

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)


```python

```

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)


```python

```

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★)


```python

```

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)


```python

```

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)


```python

```

#### 69. How to get the diagonal of a dot product? (★★★)


```python

```

#### 70. Consider the vector \[1, 2, 3, 4, 5\], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)


```python

```

#### 71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)? (★★★)


```python

```

#### 72. How to swap two rows of an array? (★★★)


```python

```

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)


```python

```

#### 74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)


```python

```

#### 75. How to compute averages using a sliding window over an array? (★★★)


```python

```

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z\[0\],Z\[1\],Z\[2\]) and each subsequent row is  shifted by 1 (last row should be (Z\[-3\],Z\[-2\],Z\[-1\]) (★★★)


```python

```

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)


```python

```

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i  (P0\[i\],P1\[i\])? (★★★)


```python

```

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P\[j\]) to each line i (P0\[i\],P1\[i\])? (★★★)


```python

```

#### 80. Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)


```python

print(R)
```

#### 81. Consider an array Z = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14\], how to generate an array R = \[\[1,2,3,4\], \[2,3,4,5\], \[3,4,5,6\], ..., \[11,12,13,14\]\]? (★★★)


```python

```

#### 82. Compute a matrix rank (★★★)


```python

```

#### 83. How to find the most frequent value in an array?




#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)


```python

```

#### 85. Create a 2D array subclass such that Z\[i,j\] == Z\[j,i\] (★★★)


```python

```

#### 86. Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)


```python

```

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)


```python

```

#### 88. How to implement the Game of Life using numpy arrays? (★★★)


```python

```

#### 89. How to get the n largest values of an array (★★★)


```python

```

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★)


```python

```

#### 91. How to create a record array from a regular array? (★★★)


```python

```

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)


```python

```

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)


```python

```

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. \[2,2,3\]) (★★★)


```python

```

#### 95. Convert a vector of ints into a matrix binary representation (★★★)


```python

```

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)


```python

```

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)


```python

```

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?


```python
# Author: Bas Swinckels


```

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)


```python

```

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)


```python

```
