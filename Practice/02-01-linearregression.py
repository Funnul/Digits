import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# 01 Parameters
learning_rate = 0.01
# epoch = one forward pass, one backward pass, of all training data
# example: 1000 trainign examples, 500 batch size
# "would take 2 iterations to complete 1 epoch"
training_epochs = 1000
# "display logs per epoch step"
display_step = 50

# 02 Training data
train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182,
                         7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27,
                         3.1])
train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596,
                         2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94,
                         1.3])

# what does this mean?
n_samples = train_X.shape[0]

# 03 tf Graph input (X is input, Y is actual??)
X = tf.placeholder("float")
Y = tf.placeholder("float")

# weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# 04 Linear model (pred)
pred = tf.add(tf.multiply(X, W), b)
# pred = (X * W) + b

# + Cost function (mean squared error)
# you want to reduce the difference of pred-Y (predicted-actual)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# + Optimizer (gradient descent(learning_rate)) while minimizing(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 05 Initialize variables (assign default values)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # run the session with init variables
    sess.run(init)

    # 01 fit training data
    # for "one round" in the range "training_epochs"
    for epoch in range(training_epochs):
        # return a zip object???
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b)

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
