from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import pandas as pd 

df = pd.read_csv('AEP_hourly.csv')
df['Year'] = df.Datetime.apply(lambda d: int(d[:4]))
df['Month'] = df.Datetime.apply(lambda d: int(d[5:7]))
df['Day'] = df.Datetime.apply(lambda d: int(d[8:10]))
df['Hour'] = df.Datetime.apply(lambda d: int(d[11:13]))
df_by_hour = df.groupby('Hour').AEP_MW.mean().reset_index()

X = np.array(df_by_hour.Hour.values).reshape(-1,1)
y = df_by_hour.AEP_MW.values

lr = LinearRegression()
lr.fit(X,y)
y_predicted = lr.predict(X)
plt.scatter(df_by_hour.Hour, df_by_hour.AEP_MW, alpha=0.6)
plt.plot( X, y_predicted)
plt.xlabel('Hours')
plt.ylabel('Megawatts avg')
plt.show()

