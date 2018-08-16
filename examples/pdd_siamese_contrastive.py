''''Script to train Siamese Neural Network on PDD data'''

from __future__ import print_function
import os
# set random seed
import numpy as np
np.random.seed(4)

from pdd.siamese import make_siamese
from pdd.applications import get_feature_extractor
from pdd.siamese.training import SiameseBatchGenerator

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
import tensorflow as tf

sess = tf.Session()
K.set_session(sess)

input_shape = (256, 256, 3)
path_to_train = "pdd_data/PDD/train"
path_to_test = "pdd_data/PDD/test"
callback_path = "pdd/applications/pretrained/siamese/e{epoch:03d}-l{val_loss:.4f}.hdf5"
os.mkdir("pdd/applications/pretrained/siamese/")


def main():
    print("Building feature extractor...")
    feature_extractor = get_feature_extractor(input_shape)

    print("Constructing siamese network...")
    siams = make_siamese(feature_extractor, dist='l2', loss='contrastive')
    siams.summary()

    print("Preparing data generator...")
    train_gen = SiameseBatchGenerator(dirname=path_to_train)
    test_gen = SiameseBatchGenerator(dirname=path_to_test)

    print("Training...")
    siams.fit_generator(
        generator=train_gen.gen(augmentation=True),
        steps_per_epoch=50,
        epochs=60,
        verbose=1,
        validation_data=test_gen.gen(),
        validation_steps=30,
        shuffle=True,
        callbacks=[
            EarlyStopping(min_delta=1e-2, patience=7, mode='min', verbose=1),
            ReduceLROnPlateau(factor=0.2, patience=5, mode='min', verbose=1),
            ModelCheckpoint(callback_path, save_weights_only=True, mode='min', verbose=1)]
    )

    print("Preparing tensorflow KNN graph")


if __name__ == '__main__':
    main()