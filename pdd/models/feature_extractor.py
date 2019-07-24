from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten


def conv_block(n_filters, 
               filter_size,
               activation='relu',
               l1_reg=0, 
               l2_reg=0, 
               dropout=0, 
               batch_norm=False):

    def _conv_block(inputs):
        # don't use bias, if batch_normalization
        bias=True if batch_norm else False

        x = Conv2D(n_filters, filter_size, use_bias=bias,
            kernel_regularizer=L1L2(l1_reg, l2_reg))(inputs)
        x = Activation(activation)(x)
        
        if batch_norm:
            x = BatchNormalization()(x)

        elif dropout > 0:
            x = Dropout(rate=dropout)(x)

        return MaxPool2D()(x)

    return _conv_block


def get_feature_extractor(input_shape):
    inputs = Input(input_shape)
    x = conv_block(32, (10, 10), batch_norm=True)(inputs)
    x = conv_block(64, (7, 7), batch_norm=True)(x)
    x = conv_block(128, (5, 5), batch_norm=True)(x)
    x = conv_block(256, (3, 3), batch_norm=True)(x)
    x = conv_block(512, (3, 3), batch_norm=True)(x)
    x = Flatten()(x)
    encoded = Dense(1024, activation='sigmoid')(x)
    return Model(inputs, encoded)