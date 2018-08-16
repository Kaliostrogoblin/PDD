import os
import numpy as np
from scipy.ndimage import imread


def read_image(filename, normalize=True, grayscale=False):
    if (grayscale):
        # one channel
        img = imread(filename, mode='L')
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    else:
        img = imread(filename)
    if normalize:
        img = img / 255.

    return img


def read_images_from_dir(dirname, **kwargs):
    img_names = os.listdir(dirname)
    imgs = [None] * len(img_names)
    for i, fname in enumerate(img_names):
        path_to_img = os.path.join(dirname, fname)
        try:
            imgs[i] = read_image(path_to_img, **kwargs)
        except:
            print("""[WARNING] 
                  Something's wrong with the file, 
                  it will be skipped""") 
    imgs = np.asarray(imgs)
    return imgs
  
  
def create_dataset_from_dir(dirname, shuffle=False, **kwargs):
    x = []
    y = []

    labels = os.listdir(dirname)                # labels are also directories

    for i, l in enumerate(labels):
        full_path = os.path.join(dirname, l)
        imgs = read_images_from_dir(full_path, **kwargs)
        x.extend(imgs)                          # add images to list
        y += [i] * len(imgs)                    # add labels to list

    x = np.asarray(x)
    y = np.asarray(y)

    if shuffle:
        idx = np.random.permutation(range(len(x)))
        x = x[idx]
        y = y[idx]

    dataset = {'data'         :  x, 
               'target'       :  y,
               'target_names' :  labels}

    return dataset