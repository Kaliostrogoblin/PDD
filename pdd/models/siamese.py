#------------------IMPORTS---------------------#
# layers 
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from ..losses import contrastive_loss
from ..distances import manhattan_distance
from ..distances import euclidean_distance
#------------------IMPORTS---------------------#


def make_siamese(twin_model, dist='l1', loss='cross_entropy', train_opt=None):
    # two inputs: left and right
    # 1: because we skip batch size
    input_shape = twin_model.layers[0].input_shape[0][1:]
    input_dtype = twin_model.layers[0].dtype
    l_input = Input(shape=input_shape, dtype=input_dtype)
    r_input = Input(shape=input_shape, dtype=input_dtype)
    # encode each of the two inputs into a vector with Char2Word model
    encoded_l = twin_model(l_input)
    encoded_r = twin_model(r_input)

    if loss == 'cross_entropy':
        loss_arg = 'binary_crossentropy'

        if dist != 'l1':
            print('%s distance is deprecated for cross entropy loss' % dist)
            print('L1 will be used')
        # merge two encoded inputs with the cosine similarity distance 
        dist = Lambda(
            manhattan_distance, 
            arguments={'elementwise': True}
        )([encoded_l, encoded_r])
        #dist = Dot(axes=-1, normalize=True)([encoded_l, encoded_r])
        # classifier on top
        output = Dense(1, activation='sigmoid')(dist)
    
    elif loss == 'contrastive':
        loss_arg = contrastive_loss

        if dist == 'l1':
            output = Lambda(
                manhattan_distance, 
                arguments={'elementwise': False}
            )([encoded_l, encoded_r])

        elif dist == 'l2':
            output = Lambda(euclidean_distance)([encoded_l, encoded_r])

        elif dist == 'cosine':
            output = Dot(axes=-1, normalize=True)([encoded_l, encoded_r])

        else:
            print("Unknown distance! Creating euclidean...")
            output = Lambda(euclidean_distance)([encoded_l, encoded_r])

    # create model
    model = Model(inputs=[l_input, r_input], outputs=output)
    # compile it
    train_opt = Adam(lr=0.0001) if train_opt is None else train_opt
    model.compile(loss=loss_arg, optimizer=train_opt, metrics=['accuracy']) 
    return model