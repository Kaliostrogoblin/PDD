'''Functions for the visualization of the some metrics
'''

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(y_true, 
                          y_pred, 
                          target_names, 
                          title='Confusion matrix', 
                          xticks_rotation=80,
                          savefig=False):
    '''Plots the cm confusion matrix
    '''
    cm = confusion_matrix(y_true, y_pred)
    # create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Blues)
    plt.title(title)
    # add colorbar
    plt.colorbar()
    # add the names of the target_names as ticks
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=xticks_rotation, fontsize=14)
    plt.yticks(tick_marks, target_names, fontsize=14)

    # colorization
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center", 
                 fontsize=12,
                 color="white" if cm[i, j] > thresh else "black")
    # additional settings
    plt.tight_layout()
    plt.grid(False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # save figure
    if savefig:
        plt.savefig('confusion_matrix.png')        
        
    plt.show()


def plot_incorrect_predictions(imgs,  
                               y_true, 
                               y_pred,
                               target_names, 
                               savefig=False):
    '''Plots the incorrect classified images
    '''
    incorrect_preds_idx = np.where(y_true != y_pred)[0]
    # the height of the figure is (number of images * width)
    # one image per row
    plt.figure(figsize=(5, len(incorrect_preds_idx)*5))

    for i, idx in enumerate(incorrect_preds_idx):
        # create subplot 
        plt.subplot(len(incorrect_preds_idx), 1, i+1)
        # get true class
        true_class = y_true[idx]
        true_class = target_names[true_class]
        # get predicted class
        pred_class = y_pred[idx]
        pred_class = target_names[pred_class]
        # plot image and add caption
        plt.title('True - `%s`, Predicted - `%s`' % (true_class, pred_class))
        plt.imshow(imgs[idx])
        # disable grid
        plt.grid(False)
        # disable ticks
        plt.xticks([]), plt.yticks([])
    # save figure
    if savefig:
        plt.savefig('incorrect_preds.png')
        
    plt.show()


def plot_embeddings(X, y, 
                    X_origin=None,
                    show_as_imgs=False, 
                    title=None, 
                    savefig=False):
    '''Scale and visualize the embedding vectors
    '''
    # extract components
    tsne = TSNE(n_components=2, perplexity=5)
    tsne_embeddings = tsne.fit_transform(X)
    # scale them
    x_min = np.min(tsne_embeddings, 0)
    x_max = np.max(tsne_embeddings, 0)
    tsne_embeddings = (tsne_embeddings - x_min) / (x_max - x_min)
    # create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(len(X)):
        # print the label of the sample
        plt.text(x = tsne_embeddings[i, 0], 
                 y = tsne_embeddings[i, 1], 
                 s = str(y[i]),
                 color=plt.cm.Set1(y[i] / np.unique(y).size),
                 fontdict={'weight': 'bold', 'size': 12})
    
    # replace text labels with original images
    if show_as_imgs and X_origin is not None:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(len(X)):
                # calculate distance between embeddings
                dist = tsne_embeddings[i] - shown_images
                dist = np.sum(dist ** 2, 1)
                # don't show points that are too close
                if np.min(dist) < 4e-4:
                    continue
                # add index of the image to the shown_images list
                shown_images = np.r_[shown_images, [tsne_embeddings[i]]]
                # plot original image
                img = offsetbox.OffsetImage(X_origin[i], 
                                            cmap=plt.cm.gray_r, 
                                            zoom=0.25, 
                                            filterrad=0.1)
                # plot img using embedding point as coords
                imagebox = offsetbox.AnnotationBbox(img, tsne_embeddings[i])
                ax.add_artist(imagebox)
    else:
        print("To annotate embeddings with the original images, X_origin is required!")
    # disable grid
    plt.grid(False)
    # disable ticks
    plt.xticks([]), plt.yticks([])
    # print title
    if title is not None:
        plt.title(title)
    # save figure
    if savefig:
        plt.savefig('tsne.png', dpi=300)
        
    plt.show()