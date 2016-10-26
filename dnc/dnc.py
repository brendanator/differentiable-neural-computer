import tensorflow as tf
from collections import namedtuple
from memory_network import MemoryNetwork

DncState = namedtuple('DncState', ('controller_state', 'memory_state', 'read_vectors'))

class DifferentiableNeuralComputer(tf.nn.rnn_cell.RNNCell):
  def __init__(self, controller_network, memory_locations, memory_width, num_read_heads):
    self.controller_network = controller_network
    self.memory_network = MemoryNetwork(memory_locations, memory_width, num_read_heads)

  @property
  def state_size(self):
    return DncState(
      self.controller_network.state_size,
      self.memory_network.state_size,
      self.memory_network.output_size)

  @property
  def output_size(self):
    return self.controller_network.output_size + self.memory_network.output_size

  def __call__(self, inputs, state):
    with tf.variable_scope('dnc'):
      batch_size = inputs.get_shape().as_list()[0]

      controller_state, memory_state, read_vectors = state
      read_vectors = tf.reshape(read_vectors, [batch_size, -1])

      controller_state = tf.check_numerics(controller_state, 'controller_state')
      # memory_state = tf.check_numerics(memory_state, 'memory_state')
      read_vectors = tf.check_numerics(read_vectors, 'read_vectors')

      controller_input = tf.concat(0, [inputs, read_vectors])
      controller_output, new_controller_state = self.controller_network(inputs, controller_state)

      new_read_vectors, new_memory_state = self.memory_network(controller_output, memory_state)

      new_read_vectors = tf.Print(new_read_vectors, [new_read_vectors], 'new_read_vectors')
      output = tf.reshape(tf.concat(1, [controller_output, new_read_vectors]), [-1, self.output_size])
      new_state = DncState(new_controller_state, new_memory_state, new_read_vectors)
      return output, new_state
