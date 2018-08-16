'''Batch generator class for Siamese text classifier
'''
# for data augmentation
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from glob import glob
from abc import ABC, abstractmethod
from scipy.ndimage import imread
import os

augmenter = ImageDataGenerator(
    rotation_range = 75,
    shear_range=0.3, 
    zoom_range=0.3, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    channel_shift_range=0.2,
    vertical_flip=True,
    horizontal_flip=True
)

def checkEqual(lst):
    '''Checks if all elements in list are equal'''
    return not lst or lst.count(lst[0]) == len(lst)


def shuffle_arrays(*args, axes=None):
    '''Axes argument is required 
    '''
    if axes is None:
        # if axes weren't pass, then compare 0-axis
        sizes = [len(x) for x in args]
    else:
        assert len(axes) == len(args), "Axes argument should have the same length as args"
        sizes = [args[i].shape[axes[i]] for i in range(len(axes))]

    assert checkEqual(sizes), "Input arrays must have same sizes"
    # permute indices
    idx = np.random.permutation(args[0].shape[axes[0]])
    return [np.take(args[i], idx, axis=axes[i]) for i in range(len(axes))]


class BaseBatchGenerator(ABC):
    '''Abstract class for Batch Generator
    '''
    @abstractmethod
    def __init__(self, X=None, y=None, batch_size=32, dirname=None):
        if dirname is not None:
            # we already have all statistics
            self._from_directory(dirname, batch_size)
        else:
            self._from_arrays(X, y, batch_size)


    @abstractmethod
    def _from_arrays(self, X, y, batch_size=32):
        pass


    @abstractmethod
    def _from_directory(self, dirname, batch_size=32):
        pass


    @abstractmethod
    def next_batch():
        pass


class SiameseBatchGenerator(BaseBatchGenerator):
    """ For loading utterances from input dataset with categories
    and making batches of pairs of images, during training
    Positive pair (1) - utterances of one class
    Negative pair (0) - utterances from different classes

    # Arguments
        X : np.array, optional[default=None]
            input samples

        y : np.array, optional[default=None]
            labels

        batch_size : int, optional[default=32]
                     the size of input batch

        dirname: string, optional[default=32]
                 path to the directory with images
    """
    def __init__(self, X=None, y=None, batch_size=32, dirname=None):
        super(SiameseBatchGenerator, self).__init__(X, y, batch_size, dirname)
            
            
    def _from_arrays(self, X, y, batch_size=32):
        assert X is not None and y is not None, "X and y input arrays are required"
        self.x = X
        self.y = y
        self.batch_size = batch_size
        self.__count_stats()


    def _from_directory(self, dirname, batch_size=32):
        '''Constructor only for images
        '''
        assert os.path.isdir(dirname), "There is no such directory `%s`" % dirname

        X, y = [], []
        class_folders = glob(os.path.join(dirname, "*", ""))

        self.dirname = dirname
        self.batch_size = batch_size
        self.n_classes = len(class_folders)
        self.samples_per_class = np.zeros(self.n_classes, dtype=np.int32)
        self.class_idx = [None]*self.n_classes

        for i, folder in enumerate(class_folders):
            img_fnames = glob(os.path.join(folder, '*.jpg'))
            # add all image files with other extensions
            for ext in ["*.png", "*jpeg"]:
                img_fnames.extend(glob(os.path.join(dirname, folder, ext)))
            # add filenames and corresponding labels to array
            X.extend(img_fnames)
            y.extend([i]*len(img_fnames))
            self.samples_per_class[i] = len(img_fnames)
            # split sorted indices on classes
            if i == 0:
                self.class_idx[i] = np.arange(len(img_fnames))
            else:
                low = sum(self.samples_per_class[:i])
                high = low + self.samples_per_class[i]
                self.class_idx[i] = np.arange(low, high, dtype=np.int32)
        # transform to arrays for convenience
        self.x = np.array(X)
        self.y = np.array(y)
        

    def __count_stats(self):
        self.samples_per_class = np.unique(self.y, return_counts=True)[1]
        self.n_classes = len(self.samples_per_class)
        # sort indices by their value, i.e. sort labels
        sorted_idx = np.argsort(self.y)
        # split sorted indices on classes
        self.class_idx = np.split(sorted_idx, np.cumsum(self.samples_per_class)[:-1])


    def __get_pair(self, c, pos):
        '''c - class number
           pos - positive or negative
        '''
        # randomly select two samples for each class to create pair
        idx = np.random.permutation(self.samples_per_class[c])[:2]
        
        if not pos or len(idx) == 1:
            # for negatives choose the opposite class 
            c_ = np.random.choice([x for x in range(self.n_classes) if x != c])
            # choose the sample from the opposite class
            i_ = np.random.randint(self.samples_per_class[c_])
            l_sample = self.x[self.class_idx[c][idx[0]]]
            r_sample = self.x[self.class_idx[c_][i_]]
            return l_sample, r_sample

        return self.x[self.class_idx[c][idx]]


    def __create_pairs(self, batch_size, pos=True):
        # if batch_size is odd number, then negatives will be one more pair
        n = (batch_size // 2) if pos else (batch_size // 2 + batch_size % 2) 
        # array for storing pairs
        pairs = np.zeros((2, n, *self.x.shape[1:]), dtype=self.x.dtype)        
        # randomly choose n class labels 
        classes = np.random.randint(self.n_classes, size=n)
        
        i = 0
        while i < n:
            pairs[0][i], pairs[1][i] = self.__get_pair(classes[i], pos)
            i += 1
        return pairs


    def __get_files_from_names(self, arr, augmentation=False):
        result = [None]*arr.size
        # read all files
        for i, x in enumerate(np.nditer(arr)):
            x = str(x)
            # normalize image
            result[i] = imread(x) / 255.
            # create random distortions
            if augmentation:
                result[i] = augmenter.random_transform(result[i])

        result = np.array(result)
        result = result.reshape((*arr.shape, *result[0].shape))
        return result


    def next_batch(self, batch_size=None, shuffle=True, seed=None, augmentation=False):
        if seed is not None:
            np.random.seed(seed)
        # if batch size was not specified use the default one
        batch_size = self.batch_size if batch_size is None else batch_size
        # arrays for pairs and labels respectively
        batch_xs = np.zeros((2, batch_size, *self.x.shape[1:]), dtype=self.x.dtype)
        batch_ys = np.ones((batch_size, ))
        # positive pairs
        batch_xs[:, :batch_size // 2] = self.__create_pairs(batch_size)
        # negative pairs
        batch_xs[:, batch_size // 2:] = self.__create_pairs(batch_size, pos=False)
        batch_ys[batch_size // 2:] = 0
        # permutation
        if shuffle:
            batch_xs, batch_ys = shuffle_arrays(batch_xs, batch_ys, axes=[1, 0])
        # if flow_from_dir = True, batch_xs - filenames
        # so we should to read files 
        if self.dirname is not None:
            batch_xs = self.__get_files_from_names(batch_xs, augmentation) 
        return [x for x in batch_xs], batch_ys


    def gen(self, batch_size=None, shuffle=True, seed=None, augmentation=False):
        if seed is not None:
            np.random.seed(seed)

        while True:
            batch_xs, batch_ys = self.next_batch(
                batch_size=batch_size, shuffle=shuffle, augmentation=augmentation)
            yield [x for x in batch_xs], batch_ys


