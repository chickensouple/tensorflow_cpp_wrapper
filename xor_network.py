import tensorflow as tf
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants

class XorNetwork(object):
    def __init__(self, sess=None):
        self._build_model()
        if sess == None:
            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run() 
        else:
            self.sess = sess

    def _build_model(self):
        self.inputs = tf.placeholder(tf.float32, (None, 2), name="inputs")

        self.layer1 = tf.contrib.layers.fully_connected(self.inputs, 
            8, 
            activation_fn=tf.nn.relu,
            scope='layer1')
        self.layer2 = tf.contrib.layers.fully_connected(self.layer1,
            4,
            activation_fn=tf.nn.relu,
            scope='layer2')

        self.outputs = tf.contrib.layers.fully_connected(self.layer2,
            1,
            activation_fn=None,
            scope='output')

        # training
        self.labels = tf.placeholder(tf.float32, (None, 1), name="labels")
        self.loss = tf.losses.absolute_difference(self.labels, self.outputs)
        self.opt = tf.train.RMSPropOptimizer(1e-3, decay=0.9).minimize(self.loss)

    def train(self, inputs, labels):
        fd = {self.inputs: inputs, self.labels: labels}
        outputs, loss, _ = self.sess.run([self.outputs, self.loss, self.opt], feed_dict=fd)
        return outputs, loss

    def predict(self, inputs):
        fd = {self.inputs: inputs}
        outputs = self.sess.run(self.outputs, feed_dict=fd)
        return outputs


if __name__ == '__main__':
    xor_inputs = np.array([[0, 0],
                           [1, 1],
                           [0, 1],
                           [1, 0]])
    xor_labels = np.array([[0, 0, 1, 1]]).T

    xn = XorNetwork()
    for i in range(2000):
        outputs, loss = xn.train(xor_inputs, xor_labels)
        print("Iteration " + str(i) + ": " + str(loss))

    outputs = xn.predict(xor_inputs)
    print("Outputs: " + str(outputs))

    # convert 
    minimal_graph = convert_variables_to_constants(xn.sess, xn.sess.graph_def, ["output/BiasAdd"])
    res = tf.train.write_graph(minimal_graph, "models/", "graph.pb", as_text=False)

