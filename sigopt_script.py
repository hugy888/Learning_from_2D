import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.python.keras.callbacks import EarlyStopping
import sigopt 
from sigopt import Connection
from tensorflow.keras.optimizers import SGD

data = sio.loadmat('dataset')
x=data['data']
y=data['label']
y=y.reshape(-1,1)
y=np.array(y)
x=x.reshape(51,51,51,5900)
x = np.transpose(x,(3,0,1,2))

X3=x[:,:,:,0]
X3=X3.reshape(5900,51,51,1)
X2=x[:,:,0,:]
X2=X2.reshape(5900,51,51,1)
X1=x[:,0,:,:]
X1=X1.reshape(5900,51,51,1)

X1_1, X1_val, y_1, y_val = train_test_split(X1, y, test_size=0.1, random_state=42)
X2_1, X2_val = train_test_split(X2, test_size=0.1, random_state=42)
X3_1, X3_val = train_test_split(X3, test_size=0.1, random_state=42)

X1_train, X1_test, y_train, y_test = train_test_split(X1_1, y_1, test_size=0.2, random_state=32)
X2_train, X2_test = train_test_split(X2_1, test_size=0.2, random_state=32)
X3_train, X3_test = train_test_split(X3_1, test_size=0.2, random_state=32)

conn = Connection(client_token="BOWNEDJATAALUDEHQLSHCNIZSRNXBHPKULVUKGIXUYEURSSM")
experiment = conn.experiments().create(
  name="multiinput_1_val",
  parameters=[
    dict(
      name="batch_size",
      bounds=dict(
        min=0,
        max=9
        ),
      type="int"
      ),

    dict(
      name="kernel_size2",
      bounds=dict(
        min=4,
        max=8
        ),
      type="int"
      ),
    dict(
      name="learning_rate",
      bounds=dict(
        min=-5,
        max=-1
        ),
      type="int"
      ),

    dict(
      name="n_filters1",
      bounds=dict(
        min=4,
        max=8
        ),
      type="int"
      ),
    dict(
      name="n_filters2",
      bounds=dict(
        min=4,
        max=8
        ),
      type="int"
      ),

    dict(
      name="mv",
      bounds=dict(
        min=1,
        max=2
        ),
      type="int"
      ),
    dict(
      name="opt",
      bounds=dict(
        min=1,
        max=2
        ),
      type="int"
      ),
    dict(
      name="n_pool",
      bounds=dict(
        min=2,
        max=8
        ),
      type="int"
      ),

    dict(
      name="n_conv",
      bounds=dict(
        min=1,
        max=3
        ),
      type="int"
      ),
    dict(
      name="units",
      bounds=dict(
        min=4,
        max=9
        ),
      type="int"
      )
    ],
  metrics=[
    dict(
      name="mae_test",
      objective="minimize",
      strategy="optimize"
      )
    ],
  observation_budget=500,
  parallel_bandwidth=1,
  project="mks",
  type="offline"
  )
print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)

def cnn(input,p):
    kernel_size2=p['kernel_size2']
    n_filters1=2**p['n_filters1']
    n_filters2=2**p['n_filters2']
    n_pool=p['n_pool']
    n_conv=p['n_conv']

    x=tf.keras.layers.Conv2D(n_filters1, (10, 10),activation="relu")(input)

    if n_conv==1:
        x=x
    elif n_conv==2:
        x=tf.keras.layers.Conv2D(n_filters2, (kernel_size2, kernel_size2),padding="same",activation="relu")(x)
    else:
        x=tf.keras.layers.Conv2D(n_filters2, (kernel_size2, kernel_size2),padding="same",activation="relu")(x)
        x=tf.keras.layers.Conv2D(n_filters2, (kernel_size2, kernel_size2),padding="same",activation="relu")(x)

    x=tf.keras.layers.AveragePooling2D((n_pool, n_pool))(x)
    x=tf.keras.layers.Flatten()(x)
    x=K.expand_dims(x,0)
    cnn=tf.keras.Model(inputs=input, outputs=x)
    return cnn

def train_model(params,X1_val,X2_val,X3_val,y_val):
    batch = 2**params['batch_size']
    lr=10**params['learning_rate']
    units=2**params['units']
    mv=params['mv']
    opt=params['opt']
    input1=tf.keras.Input(shape=(51,51,1), name="x_model")
    input2=tf.keras.Input(shape=(51,51,1), name="y_model")
    input3=tf.keras.Input(shape=(51,51,1), name="z_model")

    x=cnn(input1,params)
    y=cnn(input2,params)
    z=cnn(input3,params)

    combined = K.concatenate([x.output, y.output,z.output],0)
    if mv==1:
        combined=K.mean(combined,0)
    else:
        combined=K.max(combined,0)
    combined= tf.keras.layers.Dense(units, activation="relu")(combined)
    k= tf.keras.layers.Dense(1, activation='linear', name="final")(combined)

    model = tf.keras.Model(inputs=[x.input, y.input,z.input], outputs=k)
    sgd = SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    adam= keras.optimizers.Adam(learning_rate=lr)
    if opt==1:
        model.compile(optimizer=adam,loss="mean_absolute_error", metrics=MeanAbsolutePercentageError())
    else:
        model.compile(optimizer=sgd,loss="mean_absolute_error", metrics=MeanAbsolutePercentageError())

    early_stopping_callback = EarlyStopping(
            monitor="val_mean_absolute_percentage_error",
            min_delta=1,  # model should improve by at least 1%
            patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
            verbose=2,
            mode="min",
            restore_best_weights=True,  # restore the best model with the lowest validation error
        )

    model.fit(
        {"x_model": X1_train, "y_model": X2_train, "z_model": X3_train}, 
        {'final':y_train},
        batch_size=batch,
            epochs=100,
        validation_data=((X1_test,X2_test,X3_test), y_test),callbacks=[early_stopping_callback])

    error=model.evaluate((X1_val,X2_val,X3_val), y_val)[1]
    return error

for ii in range(500):

    suggestion = conn.experiments(experiment.id).suggestions().create()
    hparams = suggestion.assignments

    mae_test = train_model(hparams,X1_val,X2_val,X3_val,y_val)

    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        values=[dict(name='mae_test',value=mae_test)]
    )





