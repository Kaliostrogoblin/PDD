from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes, title = 'Confusion matrix', savefig=False):
    '''Plots the cm confusion matrix'''
    # create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation = 'nearest', cmap=plt.cm.Blues)
    plt.title(title)
    # add colorbar
    plt.colorbar()
    # add the names of the classes as ticks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

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


def plot_incorrect_predictions(true, predicted, savefig=False):
    incorrect_preds_idx = np.where(true != predicted)[0]
    # the height of the figure is (number of images * width)
    # one image per row
    plt.figure(figsize=(5, len(incorrect_preds_idx)*5))

    for i, idx in enumerate(incorrect_preds_idx):
        # create subplot 
        plt.subplot(len(incorrect_preds), 1, i+1)
        # get true class
        true_class = test_dataset['target'][idx]
        true_class = test_dataset['target_names'][true_class]
        # get predicted class
        pred_class = preds[idx]
        pred_class = test_dataset['target_names'][pred_class]
        # plot image and add caption
        plt.title('True - `%s`, Predicted - `%s`' % (true_class, pred_class))
        plt.imshow(test_dataset['data'][idx])
        # disable grid
        plt.grid(False)
        # disable ticks
        plt.xticks([]), plt.yticks([])
    # save figure
    if savefig:
        plt.savefig('incorrect_preds.png')
        
    plt.show()


def plot_embedding(X, y, show_as_imgs=False, title=None, savefig=False):
    '''Scale and visualize the embedding vectors'''
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
    for i in range(len(tsne_embeddings)):
        # print the label of the sample
        plt.text(x = tsne_embeddings[i, 0], 
                 y = tsne_embeddings[i, 1], 
                 s = str(y[i]),
                 color=plt.cm.Set1(y[i] / np.unique(y).size),
                 fontdict={'weight': 'bold', 'size': 12})
    
    # replace text labels with original images
    if show_as_imgs:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(len(tsne_embeddings)):
                # calculate distance between embeddings
                dist = tsne_embeddings[i] - shown_images
                dist = np.sum((dist) ** 2, 1)
                # don't show points that are too close
                if np.min(dist) < 4e-4:
                    continue
                # add index of the image to the shown_images list
                shown_images = np.r_[shown_images, [tsne_embeddings[i]]]
                # plot original image
                img = offsetbox.OffsetImage(X[i], 
                                            cmap=plt.cm.gray_r, 
                                            zoom=0.25, 
                                            filterrad=0.1)
                # plot img using embedding point as coords
                imagebox = offsetbox.AnnotationBbox(img, tsne_embeddings[i])
                ax.add_artist(imagebox)
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