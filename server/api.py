from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import flask
import io

import tensorflow as tf
import numpy as np

import sys

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
# global variable for the session
sess = None
# tensors for the input and output
input_tensor = None
n_neighbours = None
predicted_labels = None
predicted_similarities = None
# translator from class id to label
id2label = ['black_rot', 'chlorosis', 'esca', 'healthy', 'mildew']


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.
    # return the processed image
    return image


def load_model(model_path):
    global sess
    # input tensors
    global input_tensor, n_neighbours 
    # output tensors
    global predicted_labels, predicted_similarities
    # building graph
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
        model_path
        )
        
        input_tensor = graph.get_tensor_by_name("prefix/input_1:0")
        n_neighbours = graph.get_tensor_by_name('KNN_n_neighbours:0')
        predicted_labels = graph.get_tensor_by_name("predicted_classes:0")
        # we take first element, because `predicted_nearest_idx` is a tuple 
        # produced by tf.top_k operation
        predicted_similarities = graph.get_tensor_by_name("predicted_nearest_idx:0")[0]


def model_predict(inputs, k=None):
    k = 1 if k is None else k
    with sess.as_default():
        feed_dict = {
            input_tensor : inputs, 
            n_neighbours : k
        }

        preds, sims = sess.run(
            [predicted_labels, predicted_similarities], 
            feed_dict=feed_dict
        )

        preds = preds[0].tolist()
        sims = sims.tolist()

        return (preds, sims)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read number of neighbours, if less than 1, k = 1
            k = int(flask.request.args.get('n_neighbours'))
            k = max(k, 1)
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(256, 256))

            # classify the input image and then return the label
            preds, sims = model_predict(image, k=k)
            preds = [id2label[l] for l in preds]
            data["prediction"] = tuple(zip(preds, sims))
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    kwargs = dict(arg.split('=') for arg in sys.argv[1:])

    if 'model_version' in kwargs:
        model_version = int(kwargs['model_version'])
    else:
        model_version = 2

    print(("* Loading TF model and Flask starting server..."
        "please wait until server has fully started"))
    load_model("tfknn_graph/v%d" % model_version)
    app.run(debug=True, host='0.0.0.0')
