'''Batch generator class for Siamese text classifier
'''
import os
import numpy as np

from abc import ABC, abstractmethod
from glob import glob
from matplotlib.pyplot import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    def from_directory():
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
    """
    def __init__(self, X, y, 
                 batch_size=32, 
                 flow_from_dir=False, 
                 augment=False, 
                 **kwargs):

        self.x = X
        self.y = y
        self.batch_size = batch_size
        self.flow_from_dir = flow_from_dir
        self.augment = augment

        if flow_from_dir:
            # we already have all statistics
            self.__dict__.update(kwargs)
        else: 
            self.__count_stats()

        # augmentation
        if self.augment:
            self.__get_distortion_generator()


    @classmethod
    def from_directory(cls, dirname, batch_size=32, augment=False):
        '''Constructor only for images
        '''
        assert os.path.isdir(dirname), "There is no such directory `%s`" % dirname

        X, y = [], []
        class_folders = glob(os.path.join(dirname, "*", ""))

        n_classes = len(class_folders)
        samples_per_class = np.zeros(n_classes, dtype=np.int32)
        class_idx = [None]*n_classes

        for i, folder in enumerate(class_folders):
            img_fnames = glob(os.path.join(dirname, folder, '*.jpg'))
            # add all image files with other extensions
            for ext in ["*.png", "*jpeg"]:
                img_fnames.extend(glob(os.path.join(dirname, folder, ext)))
            # add filenames and corresponding labels to array
            X.extend(img_fnames)
            y.extend([i]*len(img_fnames))
            samples_per_class[i] = len(img_fnames)
            # split sorted indices on classes
            if i == 0:
                class_idx[i] = np.arange(len(img_fnames))
            else:
                low = sum(samples_per_class[:i])
                high = low + samples_per_class[i]
                class_idx[i] = np.arange(low, high, dtype=np.int32)
        # transform to arrays for convenience
        X = np.array(X)
        y = np.array(y, dtype=np.int8)
        # call __init__
        return cls(X, y, batch_size, flow_from_dir=True, 
                   augment=augment,
                   # kwargs
                   n_classes=n_classes, 
                   samples_per_class=samples_per_class, 
                   class_idx=class_idx)
            

    def __count_stats(self):
        self.samples_per_class = np.unique(self.y, return_counts=True)[1]
        self.n_classes = len(self.samples_per_class)
        # sort indices by their value, i.e. sort labels
        sorted_idx = np.argsort(self.y)
        # split sorted indices on classes
        self.class_idx = np.split(sorted_idx, np.cumsum(self.samples_per_class)[:-1])


    def __get_distortion_generator(self):
        self.distortion_generator = ImageDataGenerator(
            rotation_range = 75,
            shear_range=0.3, 
            zoom_range=0.3, 
            width_shift_range=0.2, 
            height_shift_range=0.2,
            channel_shift_range=0.2,
            vertical_flip=True,
            horizontal_flip=True
        )


    def random_distortion(self, img):
        return self.distortion_generator.random_transform(img)


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


    def __get_files_from_names(self, arr):
        result = [None]*arr.size
        # read all files
        for i, x in enumerate(np.nditer(arr)):
            result[i] = imread(str(x)) / 255.
            if self.augment:
                result[i] = self.random_distortion(result[i])
                
        result = np.array(result)
        result = result.reshape((*arr.shape, *result[0].shape))
        return result


    def next_batch(self, batch_size=None, shuffle=True, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # if batch size was not specified use the default one
        batch_size = self.batch_size if batch_size is None else batch_size
        # arrays for pairs and labels respectively
        batch_xs = np.zeros((2, batch_size, *self.x.shape[1:]), dtype=self.x.dtype)
        batch_ys = np.ones((batch_size, ), dtype=np.int8)
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
        if self.flow_from_dir:
            batch_xs = self.__get_files_from_names(batch_xs) 
        return batch_xs, batch_ys




