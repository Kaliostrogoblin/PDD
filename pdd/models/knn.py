import keras.backend as K
import tensorflow as tf


class TfKNN:
    def __init__(self, sess, feature_extractor, support_set):
        self.sess = sess
        self.make_fknn_graph(feature_extractor, support_set)


    def make_fknn_graph(self, feature_extractor, support_set):
        '''fast KNN graph
    
        # Arguments
            feature_extractor : keras pretrained model
            support_set : training set (x and y)
        '''
        K.set_learning_phase(0) # testing phase
        # rescale support_set, extracting features
        self.support_set_x = feature_extractor.predict(support_set[0])
        self.train_labels = tf.constant(support_set[1], dtype=tf.int32,
            name="support_set_y")
        # tensor for support set
        self.keys = tf.constant(self.support_set_x, dtype=tf.float32, 
            name="support_set_x")
        # input placeholder for test images
        self.inputs = tf.placeholder(tf.float32, 
            shape=[None, *support_set[0].shape[1:]],
            name="input_imgs")
        # rescale to embeddings space
        self.queries = feature_extractor(self.inputs)
        # normalized keys and query
        self.normalized_keys = tf.nn.l2_normalize(self.keys, axis=1)
        self.normalized_queries = tf.nn.l2_normalize(self.queries, axis=1)
        # result of the query (cosine distances)
        self.query_result = tf.matmul(
            a=self.normalized_keys, 
            b=tf.transpose(self.normalized_queries), 
            name="distance_matrix")
        # class predictions (nearest neighbour)
        self.preds_idx = tf.argmax(self.query_result, axis=0, 
            name="predicted_nearest_idx")
        self.preds_labels = tf.gather(self.train_labels, self.preds_idx,
            name="predicted_classes")


    def predict(self, imgs, return_dist=False):
        return_list = [self.preds_labels]
        
        if return_dist:
            return_list.append(self.query_result)

        return self.sess.run(return_list, feed_dict={self.inputs : imgs})


    def save_graph_for_serving(export_dir):
        tf.saved_model.simple_save(
            session=self.sess,
            inputs={"inputs" : self.inputs}, 
            export_dir=export_dir,
            outputs={
                "dist_matrix" : self.query_result,
                "preds_labels" : self.preds_labels})