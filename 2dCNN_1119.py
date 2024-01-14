import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.python.keras.callbacks import EarlyStopping

data = sio.loadmat('dataset')
x=data['data']
y=data['label']
y=y.reshape(-1,1)
y=np.array(y)
x=x.reshape(51,51,51,5900)
x = np.transpose(x,(3,0,1,2))

x0=x.reshape(5900,51,51,51,1)

x_0, x_val,y_0,y_val = train_test_split(x0, y,test_size=0.1, random_state=11)
x_train, x_test,y_train,y_test = train_test_split(x_0,y_0, test_size=0.2, random_state=32)

X3_train=x_train[:,:,:,0]
X2_train=x_train[:,:,0,:]
X1_train=x_train[:,0,:,:]

X3_test=x_test[:,:,:,0]
X2_test=x_test[:,:,0,:]
X1_test=x_test[:,0,:,:]

X3_val=x_val[:,:,:,0]
X2_val=x_val[:,:,0,:]
X1_val=x_val[:,0,:,:]

def cnn(input):
    x=tf.keras.layers.Conv2D(16, (10, 10),activation="relu")(input)
    x=tf.keras.layers.Conv2D(256, (4, 4),padding="same",activation="relu")(x)
    
    x=tf.keras.layers.AveragePooling2D((8, 8))(x)
    x=tf.keras.layers.Flatten()(x)
    x=K.expand_dims(x,0)
    cnn=tf.keras.Model(inputs=input, outputs=x)
    return cnn

def get_model():
    input1=tf.keras.Input(shape=(51,51,1), name="x_model")
    input2=tf.keras.Input(shape=(51,51,1), name="y_model")
    input3=tf.keras.Input(shape=(51,51,1), name="z_model")

    x=cnn(input1)
    y=cnn(input2)
    z=cnn(input3)

    combined = K.concatenate([x.output, y.output,z.output],0)
    combined=K.mean(combined,0)
    combined= tf.keras.layers.Dense(256, activation="relu")(combined)
    k= tf.keras.layers.Dense(1, activation="linear", name="final")(combined)

    model = tf.keras.Model(inputs=[x.input, y.input,z.input], outputs=k)
    lr=0.0001
    adam= keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=adam,loss="mse", metrics=MeanAbsolutePercentageError())
    return model
model=get_model()

early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_percentage_error",
        min_delta=1,  # model should improve by at least 1%
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

run=model.fit(
    {"x_model": X1_train, "y_model": X2_train, "z_model": X3_train}, 
    {'final':y_train},
    batch_size=64,
          epochs=50,
    validation_data=((X1_test,X2_test,X3_test), y_test),callbacks=[early_stopping_callback],
    workers=8)

np.savetxt('loss1119.txt',run.history['loss'])
np.savetxt('val_loss1119.txt',run.history['val_loss'])

Y_hat=model.predict([X1_val,X2_val,X3_val])
Y_hat=Y_hat.reshape(-1)
y_val=y_val.reshape(-1)

y_hat_test=model.predict([X1_test,X2_test,X3_test])
y_hat_test=y_hat_test.reshape(-1)

y_hat_train = model.predict([X1_train,X2_train,X3_train])
y_hat_train=y_hat_train.reshape(-1)

y_train=y_train.reshape(-1)
y_test=y_test.reshape(-1)

mdic = {"y_hat_train": y_hat_train, "y_hat_test": y_hat_test, "y_val": y_val, "Y_hat": Y_hat, "y_train": y_train, "y_test": y_test}

sio.savemat("mvcnn_0608.mat", mdic)

print(model.evaluate((X1_val,X2_val,X3_val), y_val)[1])

print(np.mean(np.abs(Y_hat-y_val)/y_val))

plt.scatter(np.concatenate(y_train,y_test),np.concatenate(y_hat_train,y_hat_test),c='grey')
plt.scatter(y_val,Y_hat,c='#00B8FF')

xfit = np.array([0,100])
yfit=xfit
plt.plot(xfit,yfit)
plt.xlabel('Y_test')
plt.ylabel('Y_hat')
plt.axis('square')

plt.savefig('mvcnn0608.png', dpi=300)



