import tensorflow as tf
import numpy as np
import context
from dnc import *


def random_sequence(batch_size, sequence_length, sequence_width, sequence_padding):
  sequence = np.random.randint(2, size=[batch_size, sequence_length, sequence_width])
  return np.pad(sequence, mode='constant', pad_width=[[0,0], [sequence_padding, sequence_length + sequence_padding], [0,0]])

def n_random_sequences(n, batch_size, sequence_length, sequence_width, sequence_padding):
  random_sequences = [
    random_sequence(batch_size, sequence_length, sequence_width, sequence_padding)
    for _ in range(n)
  ]
  return np.concatenate(random_sequences, axis=1)

def required_output(input_sequences, sequence_length, sequence_padding):
  total_sequence_length = input_sequences.get_shape().as_list()[1]
  return tf.pad(tf.slice(input_sequences, [0, 0, 0], [-1, total_sequence_length - (sequence_length + sequence_padding), -1]), [[0,0],[sequence_length+sequence_padding,0],[0,0]])

def accuracy(pred, y):
  correct = tf.equal(pred, y)
  return tf.reduce_mean(tf.to_float(correct))

# r = random_sequence(1, 5, 5, 1)
# print(r)
# sess = tf.InteractiveSession()
# total_sequence_length = 12
# print(required_output(r, 5, 1))
# exit()
dtype=tf.float32
batch_size = 1
sequence_length = 5
sequence_padding = 1
sequence_width = 5
num_sequences = 10
total_sequence_length = (sequence_length + sequence_padding) * 2 * num_sequences

input_shape = [batch_size, total_sequence_length, sequence_width]
input_sequences = tf.placeholder(shape=input_shape, dtype=dtype)
required_output = required_output(input_sequences, sequence_length, sequence_padding)
controller_network = SimpleFeedforwardController(20, 2, tf.nn.relu)

dnc = DifferentiableNeuralComputer(
  controller_network,
  memory_locations=2*sequence_length,
  memory_width=2*sequence_width,
  num_read_heads=1)

# dnc = tf.nn.rnn_cell.BasicLSTMCell(20)
# dnc = tf.nn.rnn_cell.MultiRNNCell([dnc] * 3)
outputs, states = tf.scan(fn=lambda a, x: dnc(x, a[1]),
                          elems=tf.transpose(input_sequences, [1, 0, 2]),
                          initializer=(tf.zeros([batch_size, dnc.output_size]), dnc.zero_state(batch_size, dtype)))
outputs = tf.transpose(outputs, [1, 0, 2])
# print(states)
# outputs, _ = tf.nn.dynamic_rnn(dnc, input_sequences, dtype=dtype)
# print(outputs)

weights = tf.get_variable('weights', [dnc.output_size, sequence_width])
bias = tf.get_variable('bias', [sequence_width])
output_sequences = tf.reshape(
  tf.matmul(tf.reshape(outputs, [-1, dnc.output_size]), weights) + bias,
  input_shape)
output = tf.maximum(tf.sign(output_sequences), 0)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_sequences, required_output))
# optimizer = tf.train.GradientDescentOptimizer(lr=0.001)
optimizer = tf.train.AdamOptimizer()
optimize = optimizer.minimize(loss)
# gradients = optimizer.compute_gradients(loss)
# optimize = optimizer.apply_gradients(gradients)
accuracy = accuracy(output, required_output)

def show_comparison(expected, actual):
  def join_values(values):
    return ''.join([str(int(value)) for value in values])

  for e, a in zip(expected, actual):
    print(join_values(e) + ' ' + join_values(a))

with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  results = []
  for step in range(10000):
    random_sequences = n_random_sequences(num_sequences, batch_size, sequence_length, sequence_width, sequence_padding)
    acc, _, r, o, os, l = session.run([accuracy, optimize, required_output, output, output_sequences, loss],
                         feed_dict={input_sequences: random_sequences})
    show_comparison(r[0,-12:,:], o[0,-12:,:])
    print('Step %d, Accuracy %f, Loss %f' % (step, acc, l))
    results.append(acc)

print(results)
