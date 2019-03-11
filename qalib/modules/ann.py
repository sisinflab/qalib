import tensorflow as tf


def train(input, output, model_dir):

    epochs = 10000

    rows = input.shape[0]
    cols = input.shape[1]

    num_output = output.shape[1]

    X = tf.placeholder(tf.float64, [None, cols])

    W1 = tf.get_variable("w1", shape=[cols, 300], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    W2 = tf.get_variable("w2", shape=[300, 200], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    W3 = tf.get_variable("w3", shape=[200, 100], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    W4 = tf.get_variable("w4", shape=[100, num_output], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    hidden1 = tf.nn.tanh(tf.matmul(X, W1))
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, W2))
    hidden3 = tf.nn.tanh(tf.matmul(hidden2, W3))
    y_pred = tf.nn.softmax(tf.matmul(hidden3, W4))

    loss = tf.losses.mean_squared_error(output, y_pred)
    train_op = tf.train.RMSPropOptimizer(0.03).minimize(loss)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)

        for i in range(epochs):
            l, _ = session.run([loss, train_op], feed_dict={X: input})
            print("Epoch {} -- Loss: {}".format(i + 1, l))

        save_path = saver.save(session, "./" + model_dir + "/model.ckpt")


def predict(input, number, model_dir):

    tf.reset_default_graph()

    rows = input.shape[0]
    cols = input.shape[1]

    num_output = number

    X = tf.placeholder(tf.float64, [None, cols])

    W1 = tf.get_variable("w1", shape=[cols, 300], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    W2 = tf.get_variable("w2", shape=[300, 200], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    W3 = tf.get_variable("w3", shape=[200, 100], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    W4 = tf.get_variable("w4", shape=[100, num_output], dtype=tf.float64,
                         initializer=tf.random_normal_initializer)

    hidden1 = tf.nn.tanh(tf.matmul(X, W1))
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, W2))
    hidden3 = tf.nn.tanh(tf.matmul(hidden2, W3))
    y_pred = tf.nn.softmax(tf.matmul(hidden3, W4))

    saver = tf.train.Saver()

    with tf.Session() as session:

        saver.restore(session, "./" + model_dir + "/model.ckpt")

        values = session.run(y_pred, feed_dict={X: input})

    return values[0]
