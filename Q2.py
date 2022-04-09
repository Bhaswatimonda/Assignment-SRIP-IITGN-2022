import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Define dimension. 
d = 2

# Set mean vector. 
m = np.array([1, 2]).reshape(2, 1)

# Set covariance function. 
K_0 = np.array([[2, 1],
                [1, 2]])


# Eigenvalues covariance function.
np.linalg.eigvals(K_0)


# Define epsilon.
epsilon = 0.0001

# Add small pertturbation. 
K = K_0 + epsilon*np.identity(d)
#  Cholesky decomposition.
L = np.linalg.cholesky(K)

np.dot(L, np.transpose(L))

# Number of samples. 
n = 10000

u = np.random.normal(loc=0, scale=1, size=d*n).reshape(d, n)
x = m + np.dot(L, u)
sns.jointplot(x=x[0], y=x[1], kind="kde", space=0);
z = np.random.multivariate_normal(mean=m.reshape(d,), cov=K, size=n)
y = np.transpose(z)
# Plot density function.
sns.jointplot(x=y[0], y=y[1], kind="kde", space=0)


z_1 = np.random.normal(loc=0, scale=1, size=n)
z = np.random.normal(loc=0, scale=1, size=n)
z_2 = np.sign(z)*z_1
sns.jointplot(x=z_1, y=z_2, kind="kde", space=0)

