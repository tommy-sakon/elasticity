from scipy.stats import binom, poisson, norm, gamma, beta
import matplotlib.pyplot as plt
import numpy as np


# 2項分布
x = 3
n = 10
p = 0.4
print(binom.pmf(x, n, p))

x = 1
n = 1
p = 0.4
print(binom.pmf(x, n, p))

def binomial_probability(x, n, p):
    return np.math.comb(n, x) * p**x * (1-p)**(n-x)

print(binomial_probability(3, 10, 0.4))

# ポアソン分布
x = 4
lambda_ = 3
print(poisson.pmf(x, lambda_))

def poisson_probability(x, lambda_):
    return np.exp(-lambda_) * lambda_**x / np.math.factorial(x)

print(poisson_probability(4, 3))

a = np.arange(0, 11)
b = poisson.pmf(a, 3)  # 平均3のポアソン分布の0~10回までのそれぞれの確率を格納
plt.plot(a, b)
plt.show()

cumsum = np.cumsum(b)  # 累積和の計算
print(cumsum)

plt.plot(a, poisson.cdf(a, 3))  # 上のcumsumはppoisでも分布関数を出力できる
plt.show()

#1変量連続分布
# 正規分布
x = np.arange(-4, 4, 0.05)
mu = 0
sigma = 1
y = norm.pdf(x, loc=mu, scale=sigma)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# ガンマ分布
x = np.arange(0, 15, 0.05)
a1 = 1
a2 = 2
a3 = 3
s1 = 1
s2 = 2

y1 = gamma.pdf(x, a1, scale=s1)
y2 = gamma.pdf(x, a2, scale=s1)
y3 = gamma.pdf(x, a3, scale=s1)
y4 = gamma.pdf(x, a1, scale=s2)
y5 = gamma.pdf(x, a2, scale=s2)

plt.plot(x, y1, label='a=1; s=0.5')
plt.plot(x, y2, label='a=2; s=0.5', linestyle='dashed')
plt.plot(x, y3, label='a=3; s=0.5', linestyle='dashdot')
plt.plot(x, y4, label='a=1; s=2', linestyle='dotted')
plt.plot(x, y5, label='a=2; s=2', linestyle='dashdot')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.show()

# ベータ分布
x = np.arange(0, 1, 0.01)
a_values = [0.5, 1, 1, 3, 10]
b_values = [0.5, 1, 3, 3, 1]
labels = ['a=0.5; b=0.5', 'a=1; b=1', 'a=1; b=3', 'a=3; b=3', 'a=10; b=1']

y_values = [beta.pdf(x, a, b) for a, b in zip(a_values, b_values)]

plt.plot(x, y_values[0], label=labels[0])
for i in range(1, len(y_values)):
    plt.plot(x, y_values[i], label=labels[i], linestyle='dashed' if i == 1 else None)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 4)
plt.show()

import numpy as np
from scipy.stats import multinomial, multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 多項分布
x = [3, 3, 4]
n = 10
probabilities = [0.3, 0.3, 0.4]
print(multinomial.pmf(x, n, probabilities))

# 多変量正規分布
mu = np.array([0, 0])
sigma = np.array([[1, 0.3], [0.3, 2]])
k = 2

x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)
x1, x2 = np.meshgrid(x1, x2)
x = np.stack((x1.flatten(), x2.flatten()), axis=-1)

Y = np.zeros_like(x1)
for i in range(len(x1)):
    for j in range(len(x2)):
        Y[i, j] = 2 * np.pi - (-k / 2) * np.linalg.det(sigma)**(-1/2) * np.exp(
            -0.5 * np.dot(np.dot((np.array([x1[i, j], x2[i, j]]) - mu).T, np.linalg.inv(sigma)), np.array([x1[i, j], x2[i, j]]) - mu))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, Y, cmap='viridis')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# 多変量正規分布の累積確率
"""
lower = [-np.inf, -np.inf]
upper = [1.0, 1.0]
result, absolute_error, status_message = multivariate_normal.mvnun(lower, upper, mu, sigma)
print(result)
print(absolute_error)
print(status_message)
"""


#購買頻度モデルの推定-ポアソンモデル
##ポアソンモデルの対数尤度関数
import numpy as np
from scipy.optimize import minimize
from scipy.special import loggamma

def poisson_log_likelihood(theta, y):
    ll = -np.sum(theta) + np.sum(y * np.log(theta)) - np.sum(loggamma(y + 1))
    return -ll  # 最大化するために符号を反転させる

# 購買データ
y = np.array([5, 0, 1, 1, 0, 3, 2, 3, 4, 2])

# 初期値
theta0 = 1

# 最尤推定
result = minimize(poisson_log_likelihood, theta0, args=(y,), method='BFGS')

# 推定結果
theta_hat = np.exp(result.x[0])  # 推定されたθの値
se = np.sqrt(np.diag(result.hess_inv))  # 標準誤差
t_value = result.x[0] / se  # t値

print("推定されたθ:", theta_hat)
print("標準誤差:", se)
print("t値:", t_value)
