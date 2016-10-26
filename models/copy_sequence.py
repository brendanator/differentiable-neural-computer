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
num_sequences = 1
lr = 0.001
total_sequence_length = (sequence_length + sequence_padding) * 2 * num_sequences

input_shape = [batch_size, total_sequence_length, sequence_width]
input_sequences = tf.placeholder(shape=input_shape, dtype=dtype)
required_output = required_output(input_sequences, sequence_length, sequence_padding)
controller_network = SimpleFeedforwardController(20, 3)

dnc = DifferentiableNeuralComputer(
  controller_network,
  memory_locations=10,
  memory_width=sequence_width,
  num_read_heads=1)

# dnc = tf.nn.rnn_cell.BasicLSTMCell(20)
# dnc = tf.nn.rnn_cell.MultiRNNCell([dnc] * 3)
outputs, _ = tf.nn.dynamic_rnn(dnc, input_sequences, dtype=dtype)

weights = tf.get_variable('weights', [dnc.output_size, sequence_width])
bias = tf.get_variable('bias', [sequence_width])
output_sequences = tf.reshape(
  tf.matmul(tf.reshape(outputs, [-1, dnc.output_size]), weights) + bias,
  input_shape)
output = tf.maximum(tf.sign(output_sequences), 0)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_sequences, required_output))
# loss = tf.Print(loss, [loss, tf.nn.sigmoid_cross_entropy_with_logits(output_sequences, required_output)], summarize=32)
lr = 0
optimizer = tf.train.GradientDescentOptimizer(lr)
# optimizer = tf.train.AdamOptimizer()
gradients = optimizer.compute_gradients(loss)
# print(gradients)
# clipped_gradients = tf.clip_by_value(gradients, -10, 10, 'clipped')
optimize = optimizer.apply_gradients(gradients)
accuracy = accuracy(output, required_output)
check = tf.add_check_numerics_ops()

# print('\n'.join([n.name for n in tf.get_default_graph().as_graph_def().node]))

with tf.Session() as session:
  session.run(tf.initialize_all_variables())
  results = []
  for steps in range(1000):
    random_sequences = n_random_sequences(num_sequences, batch_size, sequence_length, sequence_width, sequence_padding)
    acc, _, r, o, _ = session.run([accuracy, optimize, required_output, output_sequences, loss],
                         feed_dict={input_sequences: random_sequences})
    print(r[0,:12,:])
    print(o[0,:12,:])
    print(acc)
    results.append(acc)

print(results)
