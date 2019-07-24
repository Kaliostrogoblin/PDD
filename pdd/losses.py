import tensorflow.keras.backend as K


def contrastive_loss(y_true, y_pred):
    """ Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    m = 1.0   # margin
    between_class = (1-y_true) * K.square(y_pred)      # (1-Y)*(d^2)
    within_class = y_true * K.square(K.maximum(m-y_pred, 0))  # (Y) * max((margin - d)^2, 0)
    return within_class + between_class