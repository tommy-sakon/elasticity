import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sklearn.linear_model as lm
import pickle
from datetime import date, datetime
import statsmodels.api as sm

from google.colab import drive
drive.mount('/content/drive')

common_elements = [4.0,5.0, 6.0, 7.0,8.0, 9.0,
                   10.0, 11.0,12.0, 13.0, 14.0, 16.0, 17.0, 19.0, 21.0, 23.0,24.0, 27.0, 30.0, 31.0,
                   34.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 44.0, 49.0, 50.0,56.0, 57.0, 60.0, 62.0, 66.0,
                   67.0, 68.0, 72.0,74.0, 78.0, 83.0, 85.0, 87.0, 90.0, 91.0, 92.0,96.0,
                   100.0, 101.0, 102.0, 110.0, 111.0, 112.0, 113.0,116.0, 120.0, 121.0, 122.0, 123.0, 126.0, 128.0, 130.0,131.0, 132.0,
                   134.0, 135.0, 141.0, 142.0, 143.0, 145.0,149.0, 151.0, 154.0, 155.0, 156.0, 157.0, 158.0, 160.0,162.0, 164.0,
                   170.0, 172.0, 175.0, 176.0, 178.0, 180.0, 184.0, 194.0, 195.0, 198.0, 199.0,
                   203.0, 210.0, 211.0, 213.0, 226.0, 227.0, 230.0, 231.0, 232.0,
                   237.0, 238.0, 240.0, 241.0, 246.0, 250.0, 253.0, 254.0, 256.0, 261.0, 263.0,
                   269.0, 270.0, 273.0, 281.0, 282.0, 284.0, 296.0, 297.0, 298.0, 299.0,
                   300.0, 301.0, 302.0, 306.0, 307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 315.0, 316.0, 317.0, 318.0, 319.0, 320.0, 321.0,
                   322.0, 323.0, 324.0, 325.0, 326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 333.0, 334.0, 335.0, 336.0, 337.0, 338.0, 340.0,
                   341.0, 343.0, 344.0, 346.0, 347.0, 348.0, 349.0, 350.0, 353.0, 354.0, 357.0, 361.0, 362.0, 363.0, 364.0, 367.0, 368.0,
                   369.0, 371.0, 372.0, 373.0, 375.0, 376.0, 377.0, 379.0, 380.0, 383.0, 384.0, 385.0, 386.0, 387.0, 388.0, 390.0, 391.0,
                   393.0, 394.0, 395.0, 396.0, 397.0, 398.0, 399.0,
                   400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 410.0, 411.0, 412.0, 415.0, 416.0, 417.0, 419.0, 421.0,
                   422.0, 423.0, 426.0, 428.0, 429.0, 430.0, 431.0, 436.0, 437.0, 440.0, 442.0, 443.0, 445.0, 446.0, 447.0, 451.0, 452.0,
                   453.0, 454.0, 456.0, 457.0, 458.0, 459.0, 460.0, 462.0, 463.0, 464.0, 465.0, 466.0, 467.0, 468.0, 469.0, 472.0, 473.0,
                   474.0, 475.0, 477.0, 478.0, 479.0, 480.0, 481.0, 482.0, 483.0, 486.0, 488.0, 490.0, 496.0, 497.0,
                   502.0, 503.0, 507.0, 529.0, 530.0, 531.0, 532.0, 533.0, 540.0, 541.0, 546.0, 547.0, 551.0]
places = common_elements

def estimate_elasticity(grouped, places):
  places = places
  beta_dict = {}
  beta_list = []
  r2_dict = {}
  for i in places:
      tmp = grouped[grouped.index.get_level_values(0) == i]
      if len(tmp) <= 1:
        a = np.nan
        beta_dict[i] = a
        beta_list.append(a)

        continue
      tmp = tmp.reset_index()
      X = tmp[['売価', 'date_Sun', 'date_Mon', 'date_Tue', 'date_Wed', 'date_Thu', 'date_Fri', 'date_Sat']]
      y = tmp['数量/日']

      model = lm.LinearRegression()
      model.fit(X, y)

      beta_dict[i] = float(model.coef_[0])
      beta_list.append(float(model.coef_[0]))
      r2_dict[i] = float(model.score(X,y))


  return beta_dict, beta_list, r2_dict