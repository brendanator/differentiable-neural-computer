import tensorflow as tf
import numpy as np
import os
import random
import context
from dnc import *
from curriculum import learn_curriculum


def random_sequence(batch_size, sequence_length, sequence_width, repeats):
  sequence = np.random.randint(2, size=[batch_size, sequence_length, sequence_width])

  input_sequence = np.zeros([batch_size, (repeats+1) * sequence_length, sequence_width + 2])
  input_sequence[:, :sequence_length, :sequence_width] = sequence
  input_sequence[:, sequence_length-1, -2] = 1
  input_sequence[:, sequence_length, -1] = repeats

  target_sequence = np.zeros([batch_size, (repeats+1) * sequence_length, sequence_width])
  for r in range(repeats):
    target_sequence[:, sequence_length*(r+1):sequence_length*(r+2), :] = sequence

  return input_sequence, target_sequence


def random_sequences(batch_size, num_sequences, sequence_length, sequence_width, max_repeats):
  input_sequences, target_sequences = [], []
  for _ in range(num_sequences):
    repeats = random.randint(1, max_repeats)
    input_sequence, target_sequence = random_sequence(batch_size, sequence_length, sequence_width, repeats)
    input_sequences.append(input_sequence)
    target_sequences.append(target_sequence)

  input_sequences = np.concatenate(input_sequences, axis=1)
  target_sequences = np.concatenate(target_sequences, axis=1)

  return input_sequences, target_sequences


def random_sequences_lesson(batch_size, num_sequences, sequence_length, sequence_width, max_repeats):
  return lambda: random_sequences(batch_size, num_sequences, sequence_length, sequence_width, max_repeats)


if __name__ == '__main__':

  # Check checkpoint directory
  if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')

  # Options
  batch_size = 32
  sequence_width = 6
  dtype=tf.float32

  # Input/output placeholders
  input_sequences = tf.placeholder(shape=[batch_size, None, sequence_width+2], dtype=dtype)
  target_sequences = tf.placeholder(shape=[batch_size, None, sequence_width], dtype=dtype)

  # Differentiable neural computer
  controller_network = SimpleFeedforwardController(20, 2, tf.nn.relu)
  dnc = DifferentiableNeuralComputer(
    controller_network,
    memory_locations=50,
    memory_width=2*sequence_width,
    num_read_heads=1)
  dnc_output, _ = tf.nn.dynamic_rnn(
    dnc,
    input_sequences,
    dtype=dtype,
    initial_state=dnc.zero_state(batch_size, dtype))

  # Predict output
  weights = tf.get_variable('weights', [dnc.output_size, sequence_width])
  bias = tf.get_variable('bias', [sequence_width])
  output_sequences = tf.reshape(
    tf.matmul(tf.reshape(dnc_output, [-1, dnc.output_size]), weights) + bias,
    [batch_size, -1, sequence_width])
  predicted_sequences = tf.maximum(tf.sign(output_sequences), 0)

  # Loss and training operations
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_sequences, target_sequences))
  optimizer = tf.train.AdamOptimizer()
  gradients = optimizer.compute_gradients(loss)
  gradients = [(tf.clip_by_value(gradient, -10, 10), variable) for gradient, variable in gradients]
  optimize = optimizer.apply_gradients(gradients)
  accuracy = tf.reduce_mean(tf.to_float(tf.equal(predicted_sequences, target_sequences)))


  with tf.Session() as session:

    def train(input_, target):
      _, acc = session.run([optimize, accuracy],
                          feed_dict={input_sequences: input_, target_sequences: target})
      # Success on 95% accuracy
      return acc > 0.95

    def print_sequence(sequence):
      # Only print first example in batch
      for values in np.transpose(sequence[0]):
        print(''.join([str(int(value)) for value in values]))

    def evaluate(input_, target, step, lesson):
      acc, loss_, predicted = session.run([accuracy, loss, predicted_sequences],
                                          feed_dict={input_sequences: input_, target_sequences: target})

      print('Step %d, Lesson %d, Accuracy %f, Loss %f' % (step, lesson, acc, loss_))
      print('Input:')
      print_sequence(input_)
      print('Expected:')
      print_sequence(target)
      print('Predicted:')
      print_sequence(predicted)
      print('')

      if step % 1000 == 0:
        tf.train.Saver(tf.trainable_variables()).save(session, 'checkpoints/copy.model', global_step=step)


    session.run(tf.initialize_all_variables())

    lessons = [
      random_sequences_lesson(
        batch_size,
        num_sequences=3,
        sequence_length=length,
        sequence_width=sequence_width,
        max_repeats=1)
      for length in range(2,10)]

    learn_curriculum(
      lessons,
      train,
      evaluate,
      random_lesson_ratio=0,
      level_up_streak=10,
      max_steps=50000,
      eval_period=100)

    tf.train.Saver(tf.trainable_variables()).save(session, 'checkpoints/copy.model')
