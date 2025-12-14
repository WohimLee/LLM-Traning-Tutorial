## 奇异值分解（SVD，Singular Value Decomposition）

奇异值分解（SVD，Singular Value Decomposition）是线性代数中非常重要的一种矩阵分解方法，广泛应用于数据降维、推荐系统、图像压缩等领域。

### 1 SVD 的定义
设 $A$ 是一个 $m \times n$ 的实矩阵，则存在三个矩阵 $U 、 \Sigma 、 V^T$ ，使得：

$$
A=U \Sigma V^T
$$


其中：
- $U$ 是 $m \times m$ 的正交矩阵，称为 `左奇异向量矩阵`
- $\Sigma$ 是 $m \times n$ 的对角矩阵（主对角线上为非负实数，其他为零），称为奇异值知阵；
- $V$ 是 $n \times n$ 的正交矩阵，称为 `右奇异向量矩阵`
- $V^T$ 是 $V$ 的转置。


### 2 SVD 的数学原理
SVD 基于特征值分解，但它适用于非方阵。
1. 构造两个对称正定矩阵
    - $A^TA$ ：大小为 $n \times n$ ，对称，可进行特征值分解；
    -  $AA^T$ ：大小为 $m \times m$ ，也可特征值分解；

2. 求：
    - $A^T A$ 的特征值和特征向量，设为 $\lambda_i$ 和 $v_i$ ；
    - 奇异值 $\sigma_i=\sqrt{\lambda_i}$ ；
    - 对每个 $v_i$ ，计算 $u_i=\frac{1}{\sigma_i} A v_i$ 得到左奇异向量。


### 3 SVD 分解的计算步骤（手工）
假设有矩阵：

$$
A=\left[\begin{array}{ll}
3 & 1 \\
1 & 3
\end{array}\right]
$$


##### Step 1：计算 $A^T A$

$$
A^T A=\left[\begin{array}{ll}
3 & 1 \\
1 & 3
\end{array}\right]^T\left[\begin{array}{ll}
3 & 1 \\
1 & 3
\end{array}\right]=\left[\begin{array}{cc}
10 & 6 \\
6 & 10
\end{array}\right]
$$


##### Step 2：对 $A^T A$ 求特征值和特征向量
特征值：$\lambda_1=16, ~ \lambda_2=4$

对应特征向量：$v_1=\frac{1}{\sqrt{2}}[1,1]^T, \quad v_2=\frac{1}{\sqrt{2}}[1,-1]^T$

##### Step 3：求奇异值

$$
\sigma_1=\sqrt{16}=4, \quad \sigma_2=\sqrt{4}=2
$$


##### Step 4：计算 $u_i=\frac{1}{\sigma_i} \boldsymbol{A} v_i$

$$
\begin{gathered}
u_1=\frac{1}{4} A v_1=\frac{1}{4}\left[\begin{array}{ll}
3 & 1 \\
1 & 3
\end{array}\right] \cdot \frac{1}{\sqrt{2}}\left[\begin{array}{l}
1 \\
1
\end{array}\right]=\frac{1}{\sqrt{2}}\left[\begin{array}{l}
1 \\
1
\end{array}\right] \\
u_2=\frac{1}{2} A v_2=\frac{1}{2} \cdot\left[\begin{array}{ll}
3 & 1 \\
1 & 3
\end{array}\right] \cdot \frac{1}{\sqrt{2}}\left[\begin{array}{c}
1 \\
-1
\end{array}\right]=\frac{1}{\sqrt{2}}\left[\begin{array}{c}
1 \\
-1
\end{array}\right]
\end{gathered}
$$


##### Step 5：构造 $U, \Sigma, V^T$

$$
U=\left[\begin{array}{cc}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{array}\right], \quad \Sigma=\left[\begin{array}{ll}
4 & 0 \\
0 & 2
\end{array}\right], \quad V^T=\left[\begin{array}{cc}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{array}\right]
$$

### 4 用 Python 进行 SVD（推荐）

```py
import numpy as np

A = np.array([[3, 1], [1, 3]])
U, S, VT = np.linalg.svd(A)

print("U =", U)
print("S =", S)
print("VT =", VT)
```
注意：S 是一个向量（只包含非零奇异值），你可以通过：
```py
Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, S)
```