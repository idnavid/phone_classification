
from kaldi_tools import *
from timit_tools import *
import tf_tools
import tensorflow as tf
import numpy as np


if __name__=='__main__':

    # Read TIMIT data
    root = "/Users/navidshokouhi/Software_dir/kaldi/egs/timit/feature_extraction/"
    x_trn,y_trn,x_tst,y_tst = gen_data(root)

    # tf configuration
    seed = 128
    rng = np.random.RandomState(seed)
    num_input_units = 39*10
    num_hidden_units = 250
    num_output_units = 61
    epochs = 2
    learning_rate = 0.01

    # Feed-Forward neurla network
    x,y,weights,biases,output_layer,cross_entropy,train_step = tf_tools.mlp_nn(num_input_units,
                                                                               num_output_units,
                                                                               num_hidden_units,
                                                                               learning_rate,seed)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Train
        for epoch in range(epochs):
            avg_cost = 0
            for i in x_trn:
                batch_x = x_trn[i]
                batch_y = y_trn[i]
                _, c = sess.run([train_step,cross_entropy], feed_dict={x:batch_x,y:batch_y})
                avg_cost += c*0.1/len(x_trn)
            print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)


        # Validation
        pred_temp = tf.cast(tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1)),"float")
        accuracy = tf.reduce_sum(1 - pred_temp)

        correct_frames = 0
        N = 0
        for i in x_tst:
            batch_x = x_tst[i]
            batch_y = y_tst[i]
            correct_frames += accuracy.eval({x:batch_x,y:batch_y})
            N += batch_y.shape[0]
        print correct_frames/(1.0*N)





