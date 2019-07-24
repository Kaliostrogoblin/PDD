import tensorflow.keras.backend as K


def manhattan_distance(inputs, elementwise=True):
    x, y = inputs
    if elementwise:
        return K.abs(x - y)
    else:
        return K.sum(K.abs(x - y), axis=-1, keepdims=True)


def euclidean_distance(inputs):
    x, y = inputs
    return K.sqrt(
                K.maximum(
                    K.sum(K.square(x - y), axis=-1, keepdims=True), 
                    K.epsilon())
            )