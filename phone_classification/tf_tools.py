import numpy as np
import tensorflow as tf

def dense_to_one_hot(labels_dense, num_classes=61):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    for i in range(num_labels):
        labels_one_hot[i][int(labels_dense[i])]=1
    return labels_one_hot

def create_context(feat_vec,context=9):
    feat = []
    for r in range(-1*context/2,context/2 + 1):
        feat_tmp = feat_vec
        feat_tmp = np.roll(feat_tmp,r)
        if len(feat)>0:
            feat = np.concatenate((feat,feat_tmp),axis=1)
        else:
            feat = feat_tmp
    return feat

def standard_array(feat_dict, label_dict, phone_map):
    for i in feat_dict:
        # Create context
        feat = create_context(feat_dict[i])
        feat_dict[i] = feat
        
        # segment to labels
        label = np.zeros((feat.shape[0],1))
        s = label_dict[i][0]
        e = label_dict[i][1]
        l = label_dict[i][2]
        for n in range(len(l)):
            label[s[n]:e[n]+1] = phone_map[l[n]]
        data_shape = (len(label),1)
        label_dict[i] = dense_to_one_hot(np.array(label).reshape(data_shape),num_classes=len(phone_map))
    
    return feat_dict, label_dict

def mlp_nn(num_input_units,num_output_units,num_hidden_units,learning_rate,seed):
    x = tf.placeholder(tf.float32, shape=[None, num_input_units])
    y = tf.placeholder(tf.float32, shape=[None, num_output_units])
    
    weights = {
        'hidden': tf.Variable(tf.random_normal([num_input_units, num_hidden_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([num_hidden_units, num_output_units], seed=seed))
    }
    
    biases = {
        'hidden': tf.Variable(tf.random_normal([num_hidden_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([num_output_units], seed=seed))
    }

    h = tf.add(tf.matmul(x,weights['hidden']) , biases['hidden'])
    h = tf.nn.relu(h)

    output_layer = tf.add(tf.matmul(h,weights['output']),biases['output'])
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer))
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    
    return x,y,weights,biases,output_layer,cross_entropy,train_step
