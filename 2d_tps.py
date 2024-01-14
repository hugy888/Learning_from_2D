import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from pymks import (
    PrimitiveTransformer,
    TwoPointCorrelation)

data = sio.loadmat('dataset')
x=data['data']
y=data['label']
y=y.reshape(-1,1)
y=np.array(y)
x=x.reshape(51,51,51,5900)
x = np.transpose(x,(3,0,1,2))

X=x[:,0,:,:]
Y=x[:,:,0,:]
Z=x[:,:,:,0]

data_disc_x = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(X)
data_disc_y = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(Y)
data_disc_z = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(Z)

data_corr_x = TwoPointCorrelation(
    periodic_boundary=True,
    cutoff=25,
    correlations=[(1, 1)]
).transform(data_disc_x)

data_corr_y = TwoPointCorrelation(
    periodic_boundary=True,
    cutoff=25,
    correlations=[(1, 1)]
).transform(data_disc_y)

data_corr_z = TwoPointCorrelation(
    periodic_boundary=True,
    cutoff=25,
    correlations=[(1, 1)]
).transform(data_disc_z)

data_corr_x_v=data_corr_x.reshape(5900,-1)
data_corr_y_v=data_corr_y.reshape(5900,-1)
data_corr_z_v=data_corr_z.reshape(5900,-1)

data_corr_all=np.concatenate((data_corr_x_v,data_corr_y_v,data_corr_z_v),axis=1)
data_corr_train, data_corr_test,y_train, y_test = train_test_split(data_corr_all, y,test_size=0.1, random_state=11)

pc_scores = PCA(
    svd_solver='full',
    n_components=13,
    random_state=10
).fit(data_corr_train)

pc_scores_train = pc_scores.transform(data_corr_train)
pc_scores_test = pc_scores.transform(data_corr_test)

pc_scores_train=np.array(pc_scores_train)
pc_scores_test=np.array(pc_scores_test)
poly = PolynomialFeatures(3)
poly_features = poly.fit_transform(pc_scores_train)
model = LinearRegression().fit(poly_features, y_train)
poly_features_test = poly.fit_transform(pc_scores_test)

Y_hat = model.predict(poly_features_test)
y_hat_train = model.predict(poly_features)

plt.scatter(y_train,y_hat_train,c='grey')
plt.scatter(y_test,Y_hat,c='#00B8FF')
xfit=[0,100]
yfit=xfit
plt.plot(xfit,yfit)
plt.axis('square')
plt.savefig('2d_mks0608.png', dpi=300)

print(np.mean(np.abs(Y_hat-y_test)/y_test))

mdic = {"y_hat_train": y_hat_train, "Y_hat_test": Y_hat,"y_test": y_test,"y_train": y_train}
sio.savemat("2dmks_0608.mat", mdic)