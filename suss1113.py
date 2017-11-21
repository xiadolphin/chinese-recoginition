from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tempfile

#########################      10类   ###############################################
tf.logging.set_verbosity(tf.logging.INFO)


def _parse_function(example_proto):
    features = {"img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 32, 32, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([8 * 8 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 7 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 7])
        b_fc2 = bias_variable([7])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(unused_argv):
    # Load training and eval
    filenames = "data_gray.tfrecords"
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(50)
    dataset = dataset.shuffle(buffer_size=3000)
    batched_dataset = dataset.batch(50)
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    image_train = tf.decode_raw(next_element['img_raw'], tf.uint8)
    image_train = tf.reshape(image_train, [-1, 1024])
    # print(image)
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label_train = tf.cast(next_element['label'], tf.int64)
    label_train = tf.one_hot(label_train, 7)
    x = tf.placeholder(tf.float32, [None, 1024])
    y_ = tf.placeholder(tf.float32, [None, 7])
    y_conv, keep_prob = deepnn(x)
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    f = open("train.npy", "rb")
    imageTest = np.load(f)
    imageTest = imageTest.reshape([-1, 1024])
    labelTest = np.load(f)
    #labelTest = tf.one_hot(labelTest, 7)
    f.close()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            img_train, lab_train = sess.run([image_train, label_train])
            print(sess.run(cross_entropy, feed_dict={x: img_train, y_: lab_train, keep_prob: 1.0}))
            if i % 100 == 0:
                #labelTest = sess.run([labelTest])
                train_accuracy = accuracy.eval(feed_dict={
                    x: imageTest, y_: labelTest, keep_prob: 1.0})
                print('step %d, validating accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: img_train, y_: lab_train, keep_prob: 0.5})



if __name__ == "__main__":  # 使用这种方式保证了，如果此文件被其它文件import的时候，不会执行main中的代码
    tf.app.run()    # 解析命令行参数，调用main函数 main(sys.argv)
