import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

# W = 5, -3
W = tf.Variable(-3.0)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
	print(step, sess.run(W))
	sess.run(train)