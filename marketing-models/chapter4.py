import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# データを定義する
Y = np.array([138, 48, 147, 55, 4, 21, 109, 63, 113, 85, 30, 13, 6, 2, 15, 137, 110, 13, 5, 10])
X1 = np.array([305, 398, 296, 298, 409, 404, 307, 319, 302, 398, 354, 445, 445, 445, 404, 301, 298, 398, 407, 445])
X2 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0])

# モデルを構築する
logY = np.log(Y)
logX1 = np.log(X1)
X = np.column_stack((logX1, X2))
X = sm.add_constant(X)  # 定数項を追加する
model = sm.OLS(logY, X)
results = model.fit()

# モデルの概要を表示する
print(results.summary())

# モデルの係数を表示する
print(results.params)

# 特別陳列の弾力性を計算する
elasticity_X2 = np.exp(results.params[3]) - 1
print(elasticity_X2)

#最尤法による推定
# データを定義する
logX1 = np.log(X1)
X = np.column_stack((np.ones(20), logX1, X2))  # 切片を追加する
logY = np.log(Y)

# モデルの推定を行う
betas = np.linalg.solve(X.T @ X, X.T @ logY)  # 最尤推定値を計算する
residuals = logY - X @ betas  # 残差を計算する
sigmas = np.sqrt((1 / 20) * np.sum(residuals**2))  # 標準偏差を計算する
adjusted_sigmas = np.sqrt((sigmas**2 * 20) / 17)  # 最尤法のσから二乗法のσを計算する

# t値を計算する
infoM = (1 / sigmas) * (X.T @ X)  # 情報行列を計算する
tvals = betas / np.sqrt(np.diag(np.linalg.inv(infoM)))  # t値を計算する

print(adjusted_sigmas)
print(tvals)

############################################
#クロスセクション分析

# データを定義する
Ya = np.array([138, 48, 147, 55, 4, 21, 109, 63, 113, 85, 30, 13, 6, 2, 15, 137, 110, 13, 5, 10])
Xa1 = np.array([305, 398, 296, 298, 409, 404, 307, 319, 302, 398, 354, 445, 445, 445, 404, 301, 298, 398, 407, 445])
Xa2 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0])
Yb = np.array([18, 39, 33, 114, 78, 82, 18, 120, 95, 63, 79, 85, 77, 43, 158, 9, 11, 64, 79, 84])
Xb1 = np.array([419, 398, 398, 368, 398, 398, 461, 331, 394, 368, 368, 368, 368, 380, 340, 475, 475, 388, 368, 368])
Xb2 = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

# 対数化
logYa = np.log(Ya)
logYb = np.log(Yb)
logXa1 = np.log(Xa1)
logXb1 = np.log(Xb1)

# データを結合する
T = len(Ya)
zeros = np.zeros(T)
ones = np.ones(T)
Y = np.concatenate((logYa, logYb))
X1 = np.concatenate((ones, zeros))
X2 = np.concatenate((zeros, ones))
X3 = np.concatenate((logXa1, zeros))
X4 = np.concatenate((zeros, logXb1))
X5 = np.concatenate((logXb1, logXa1))
X6 = np.concatenate((Xa2, zeros))
X7 = np.concatenate((zeros, Xb2))
X8 = np.concatenate((Xb2, Xa2))
X = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8))

# モデルを構築する
model_cross = sm.OLS(Y, X)
results_cross = model_cross.fit()

# モデルの概要を表示する
print(results_cross.summary())

############################################
#MNLモデルでのシェアモデル推定

import numpy as np
import matplotlib.pyplot as plt

beta = -2
x = np.arange(0.01, 2, 0.01)  # マーケティング変数
s = np.exp(beta * x) / (0.1 + np.exp(beta * x))  # 片方の魅力度=0.1と仮定
e = beta * (1 - s) * x

plt.plot(x, e, type='l')
plt.xlabel('x')
plt.ylabel(r'$\eta$')
plt.show()

# ----- p.63-p.65 -----
# シェアモデルの推定
price_a = np.array([305, 398, 296, 298, 409, 404, 307, 319, 302, 398, 354, 445, 445, 445, 404, 301, 298, 398, 407, 445])
price_b = np.array([229, 245, 268, 268, 268, 235, 229, 229, 229, 246, 268, 268, 268, 268, 237, 229, 229, 229, 238, 204])
price_c = np.array([475, 419, 398, 398, 358, 398, 398, 461, 301, 384, 368, 368, 368, 368, 380, 300, 475, 475, 388, 368])

ya = np.array([138, 48, 147, 55, 4, 21, 109, 63, 113, 85, 30, 13, 6, 2, 15, 137, 110, 13, 5, 10])
yb = np.array([44, 37, 22, 28, 21, 52, 61, 52, 33, 27, 26, 20, 24, 15, 29, 38, 69, 55, 16, 93])
yc = np.array([14, 32, 44, 24, 66, 29, 18, 11, 53, 29, 27, 33, 26, 31, 17, 78, 8, 10, 16, 26])

T = len(ya)
zeros = np.zeros(T)
ones = np.ones(T)
X1 = np.concatenate((ones, zeros))
X2 = np.concatenate((zeros, ones))

lprice_a = np.log(price_a)
lprice_b = np.log(price_b)
lprice_c = np.log(price_c)
lprice = np.column_stack((lprice_a, lprice_b, lprice_c))
lprice_bar = np.mean(lprice, axis=1)  # 行ごとの平均を求める

# 価格の対数中央化
X3 = np.column_stack((lprice_a - lprice_bar, lprice_b - lprice_bar))
X = np.column_stack((X1, X2, X3))

ysum = ya + yb + yc
S = np.column_stack((ya / ysum, yb / ysum, yc / ysum))
logS = np.log(S)
logS_bar = np.mean(logS, axis=1)

# シェアの対数中央化
Y = np.column_stack((logS[:, 0] - logS_bar[0], logS[:, 1] - logS_bar[1]))
model = LinearRegression(fit_intercept=False)
model.fit(X, Y)

# モデルの概要を表示する
print(model.coef_)

# シェアの予測
b0a = model.coef_[0][0]
b0b = model.coef_[0][1]
b0c = -(b0a + b0b)
b1 = model.coef_[0][2]

# 価格は中央値とする
pr_a = np.median(price_a)
pr_b = np.median(price_b)
pr_c = np.median(price_c)

Ab0a = np.exp(b0a) * pr_a**b1
Ab0b = np.exp(b0b) * pr_b**b1
Ab0c = np.exp(b0c) * pr_c**b1
sumA = Ab0a + Ab0b + Ab0c
S_hat = np.array([Ab0a, Ab0b, Ab0c]) / sumA

print(S_hat)

#############################################
#バスモデルによる推定
# 観測年と普及率のデータを定義する
Year = np.arange(1959, 2003)
W = np.array([1.6, 2, 2.7, 3.3, 3.7, 2.6, 3.4, 4.2, 4.8, 5.2, 6.1, 6.8, 7.3, 8.6, 9.7, 10.2, 11.8, 12.2, 13, 14.9, 15.5, 15.8, 16.7, 18, 17.4, 17.6, 18.3, 19.2, 20.9, 19.9, 21.9, 22.7, 23.3, 23.3, 23.2, 23.3, 22.2, 22, 22.3, 22.3, 22.9, 21.4, 22.8, 23.6])
n = len(W)
period = np.arange(1, n + 1)

# バスモデルの定義
def bass_model(period, p, q, m):
    return m * (1 - np.exp(-(p + q) * period)) / (1 + q / p * np.exp(-(p + q) * period))

# バスモデルのパラメータの推定
initial_guess = [0.01, 0.01, np.max(W)]
params, params_covariance = curve_fit(bass_model, period, W, p0=initial_guess)

# 推定したパラメータによる普及率の予測
p_hat, q_hat, m_hat = params
W_hat = m_hat * (1 - np.exp(-(p_hat + q_hat) * period)) / (1 + q_hat / p_hat * np.exp(-(p_hat + q_hat) * period))

# プロット
plt.plot(Year, W_hat, label='predicted', linestyle='-', linewidth=2)
plt.plot(Year, W, label='observed', linestyle='--', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Diffusion rate(%)')
plt.legend(loc='upper left')
plt.show()

# 推定結果の表示
print("Estimated parameters (p, q, m):", params)
