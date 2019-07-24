from ..utils.timing import timeit
from ..utils.graph_utils import freeze_session
from ..utils.graph_utils import load_frozen_graph

import tensorflow.keras.backend as K
import tensorflow as tf
import os


class TfKNN:
    def __init__(self, sess, feature_extractor, support_set):
        self.feature_extractor = feature_extractor
        self._get_keys_from_support_set(support_set[0])
        # after getting features we can freeze graph 
        self._freeze_feature_extractor_graph(sess)
        self.make_fknn_graph(support_set[1])


    @timeit
    def _get_keys_from_support_set(self, x):
        print("Getting keys from support set...")
        self.support_set_x = self.feature_extractor.predict(x)


    @timeit
    def _freeze_feature_extractor_graph(self, sess):
        print("Freezing feature extractor graph...")
        o_names = [out.op.name for out in self.feature_extractor.outputs]
        self.graph = freeze_session(sess, output_names=o_names)
       
        tmp_fname = 'feature_extractor.pb'
        tf.train.write_graph(self.graph, "", tmp_fname, as_text=False)

        tf.reset_default_graph()
        self.graph = load_frozen_graph(tmp_fname)
        os.remove(tmp_fname)


    def _prepare_support_set(self, labels):
        # rescale support_set, extracting features
        self.keys = tf.constant(self.support_set_x, dtype=tf.float32, 
            name="support_set_x")
        self.support_set_y = tf.constant(labels, 
            dtype=tf.int32, name="support_set_y")


    @timeit
    def make_fknn_graph(self, labels):
        '''fast KNN graph
    
        # Arguments
            labels : training labels
        '''
        print("Creating TfKNN graph...")
        self.inputs = self.graph.get_tensor_by_name('prefix/input_1:0')
        # rescale to embeddings space
        self.inputs_rescaled = self.graph.get_tensor_by_name('prefix/dense_1/Sigmoid:0')
        
        with self.graph.as_default():
            self._prepare_support_set(labels)
            # init placeholder for the number of neighbours
            self.n_neighbours = tf.placeholder(tf.int32, name="KNN_n_neighbours")
            # input placeholder for test images
            # self.norm_inputs = tf.cond(
            #     tf.reduce_max(self.inputs) > 1, 
            #     lambda: self.inputs / 255., # if True, normalize
            #     lambda: self.inputs) # else use the original ones
            # normalized keys and query
            self.normalized_keys = tf.nn.l2_normalize(self.keys, axis=1)
            self.normalized_queries = tf.nn.l2_normalize(self.inputs_rescaled, axis=1)
            # result of the query (cosine distances)
            self.query_result = tf.matmul(
                a=self.normalized_keys, 
                b=tf.transpose(self.normalized_queries), 
                name="distance_matrix")
            # predict similarities and indices for n_neighbours
            self.preds_sims, self.preds_idx = tf.nn.top_k(
                tf.transpose(self.query_result), 
                k=self.n_neighbours, 
                name="predicted_nearest_idx"
            )
            # predicted classes
            self.preds_labels = tf.gather(self.support_set_y, self.preds_idx,
                name="predicted_classes")

    @timeit
    def predict(self, imgs, n_neighbours=1):
        print("Making prediction for %d images..." % len(imgs))
        with tf.Session(graph=self.graph) as sess:
            return sess.run([self.preds_labels, self.preds_sims], 
                            feed_dict={
                                self.inputs : imgs, 
                                self.n_neighbours : n_neighbours
                            })


    @timeit
    def save_graph_for_serving(self, export_dir):
        print("Saving graph for serving...")
        with tf.Session(graph=self.graph) as sess:
            tf.saved_model.simple_save(
                session=sess,
                inputs={
                    "inputs" : self.inputs,
                    "n_neighbours" : self.n_neighbours}, 
                export_dir=export_dir,
                outputs={
                    "predicted_labels" : self.preds_labels,
                    "predicted_similarities" : self.preds_sims})