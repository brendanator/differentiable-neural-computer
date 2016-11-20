import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import with_dependencies
import collections

MemoryNetworkState = collections.namedtuple('MemoryNetworkState', ('memory', 'usage', 'precedence_weighting', 'temporal_linkage', 'read_weightings', 'write_weighting'))

class MemoryNetwork(tf.nn.rnn_cell.RNNCell):
  def __init__(self, memory_locations, memory_width, num_read_heads):
    self.memory_locations = memory_locations
    self.memory_width = memory_width
    self.num_read_heads = num_read_heads

  @property
  def state_size(self):
    return MemoryNetworkState(
      memory = tf.TensorShape([self.memory_locations, self.memory_width]),
      usage = self.memory_locations,
      precedence_weighting = self.memory_locations,
      temporal_linkage = tf.TensorShape([self.memory_locations, self.memory_locations]),
      read_weightings = tf.TensorShape([self.memory_locations, self.num_read_heads]),
      write_weighting = self.memory_locations)

  @property
  def output_size(self):
    return self.num_read_heads * self.memory_width

  def set_memory_locations(memory_locations):
    """Once the memory network has been trained it should work with a larger memory"""
    self.memory_locations = memory_locations

  def __call__(self, inputs, state):
    with tf.variable_scope('MemoryNetwork'):
      # Extract inputs properties
      batch_size = inputs.get_shape().as_list()[0]
      dtype = inputs.dtype

      # Extract state
      memory, usage, precedence_weighting, temporal_linkage, read_weightings, write_weighting = state

      # Initiase variables
      read_heads = ReadHeads(inputs, self.num_read_heads, self.memory_width, dtype)
      write_head = WriteHead(inputs, self.memory_width, dtype)

      # Dynamic memory allocation
      with tf.variable_scope('memory_allocation'):
        memory_retention = tf.reduce_prod(1 - read_heads.free_gate * read_weightings, 2)
        new_usage = (usage + write_weighting - usage * write_weighting) * memory_retention
        sorted_usage, indices = tf.nn.top_k(-new_usage, k=self.memory_locations, sorted=True)
        sorted_usage = -sorted_usage
        scalings = tf.cumprod(sorted_usage, 1, exclusive=True)
        allocation_weighting = (1 - sorted_usage) * scalings
        # Due to tf.gather_nd not implementing gradients yet a loop is needed :(
        allocation_weighting = tf.pack(
          [tf.gather(allocation_weighting[batch], indices[batch, :]) for batch in range(batch_size)],
          name='allocation_weighting'
        )

      # Write weighting
      with tf.variable_scope('write_weighting'):
        write_content_weighting = tf.squeeze(self.content_weighting(memory, write_head.write_key, write_head.write_strength))
        new_write_weighting = write_head.write_gate * (
          write_head.allocation_gate * allocation_weighting
          + (1 - write_head.allocation_gate)
          * write_content_weighting)

      # Temporal memory linkage
      with tf.variable_scope('temporal_linkage'):
        write_weighting_matrix = tf.tile(tf.expand_dims(new_write_weighting, 2), [1, 1, self.memory_locations])
        new_temporal_linkage = ((1 - write_weighting_matrix - tf.transpose(write_weighting_matrix, [0, 2, 1])) * temporal_linkage
                                + tf.batch_matmul(tf.expand_dims(new_write_weighting, 2), tf.expand_dims(precedence_weighting, 1)))
        non_diagonal_ones = tf.matrix_set_diag(tf.ones([batch_size, self.memory_locations, self.memory_locations]), tf.zeros([batch_size, self.memory_locations]))
        new_temporal_linkage = new_temporal_linkage * non_diagonal_ones

        new_precedence_weighting = (1 - tf.reduce_sum(new_write_weighting, reduction_indices=1, keep_dims=True)) * precedence_weighting + new_write_weighting

      # Write memory
      with tf.variable_scope('write_memory'):
        new_memory = (
            memory * (1 - tf.batch_matmul(tf.expand_dims(new_write_weighting, 2), tf.expand_dims(write_head.erase_vector, 1)))
            + tf.batch_matmul(tf.expand_dims(new_write_weighting, 2), tf.expand_dims(write_head.write_vector, 1)))

      # Read memory
      with tf.variable_scope('read_memory'):
        backward_weighting = tf.batch_matmul(new_temporal_linkage, read_weightings, adj_x=True)
        read_content_weighting = self.content_weighting(new_memory, read_heads.read_key, read_heads.read_strength)
        forward_weighting = tf.batch_matmul(new_temporal_linkage, read_weightings)

        bcf_weightings = tf.pack([backward_weighting, read_content_weighting, forward_weighting], 2)
        new_read_weighting = tf.reduce_sum(read_heads.read_mode * bcf_weightings, reduction_indices=2, name='new_read_weighting')

        read_result = tf.batch_matmul(new_memory, new_read_weighting, adj_x=True)

      # Return results
      output = tf.reshape(read_result, [batch_size, self.num_read_heads * self.memory_width])
      new_state = MemoryNetworkState(new_memory, new_usage, new_precedence_weighting, new_temporal_linkage, new_read_weighting, new_write_weighting)
      return output, new_state

  def content_weighting(self, memory, key, strength):
    with tf.variable_scope('content_weighting'):
      normalised_memory = tf.nn.l2_normalize(memory, 2)
      normalised_key = tf.nn.l2_normalize(key, 1)

      similarity = tf.batch_matmul(normalised_memory, normalised_key)

      return tf.nn.softmax(similarity * strength, 1)

class ReadHeads(object):
  def __init__(self, inputs, num_heads, memory_width, dtype):
    inputs_len = inputs.get_shape().as_list()[1]

    with tf.variable_scope('ReadHeads'):
      read_key_weights = tf.get_variable('read_key_weights', shape=[inputs_len, memory_width * num_heads], dtype=dtype)
      read_key_bias = tf.get_variable('read_key_bias', shape=[memory_width * num_heads], dtype=dtype)
      self.read_key = tf.reshape(tf.nn.xw_plus_b(inputs, read_key_weights, read_key_bias), [-1, memory_width, num_heads], name='read_key')

      read_strength_weights = tf.get_variable('read_strength_weights', shape=[inputs_len, num_heads], dtype=dtype)
      read_strength_bias = tf.get_variable('read_strength_bias', shape=[num_heads], dtype=dtype)
      self.read_strength = one_plus(tf.nn.xw_plus_b(inputs, read_strength_weights, read_strength_bias), name='read_strength')

      free_gate_weights = tf.get_variable('free_gate_weights', shape=[inputs_len, num_heads], dtype=dtype)
      free_gate_bias = tf.get_variable('free_gate_bias', shape=[num_heads], dtype=dtype)
      self.free_gate = tf.expand_dims(tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, free_gate_weights, free_gate_bias)), 1, name='free_gate')

      read_mode_weights = tf.get_variable('read_mode_weights', shape=[inputs_len, 3 * num_heads], dtype=dtype)
      read_mode_bias = tf.get_variable('read_mode_bias', shape=[3 * num_heads], dtype=dtype)
      self.read_mode = tf.nn.softmax(tf.reshape(tf.nn.xw_plus_b(inputs, read_mode_weights, read_mode_bias), [-1, 1, 3, num_heads]), 2, name='read_mode')

class WriteHead(object):
  def __init__(self, inputs, memory_width, dtype):
    inputs_len = inputs.get_shape().as_list()[1]

    with tf.variable_scope('WriteHead'):
      write_key_weights = tf.get_variable('write_key_weights', shape=[inputs_len, memory_width], dtype=dtype)
      write_key_bias = tf.get_variable('write_key_bias', shape=[memory_width], dtype=dtype)
      self.write_key = tf.expand_dims(tf.nn.xw_plus_b(inputs, write_key_weights, write_key_bias), 2, name='write_key')

      write_strength_weights = tf.get_variable('write_strength_weights', shape=[inputs_len, 1], dtype=dtype)
      write_strength_bias = tf.get_variable('write_strength_bias', shape=[1], dtype=dtype)
      self.write_strength = tf.expand_dims(one_plus(tf.nn.xw_plus_b(inputs, write_strength_weights, write_strength_bias)), 2, name='write_strength')

      erase_vector_weights = tf.get_variable('erase_vector_weights', shape=[inputs_len, memory_width], dtype=dtype)
      erase_vector_bias = tf.get_variable('erase_vector_bias', shape=[memory_width], dtype=dtype)
      self.erase_vector = tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, erase_vector_weights, erase_vector_bias), name='erase_vector')

      write_vector_weights = tf.get_variable('write_vector_weights', shape=[inputs_len, memory_width], dtype=dtype)
      write_vector_bias = tf.get_variable('write_vector_bias', shape=[memory_width], dtype=dtype)
      self.write_vector = tf.nn.xw_plus_b(inputs, write_vector_weights, write_vector_bias, name='write_vector')

      allocation_gate_weights = tf.get_variable('allocation_gate_weights', shape=[inputs_len, 1], dtype=dtype)
      allocation_gate_bias = tf.get_variable('allocation_gate_bias', shape=[1], dtype=dtype)
      self.allocation_gate = tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, allocation_gate_weights, allocation_gate_bias), name='allocation_gate')

      write_gate_weights = tf.get_variable('write_gate_weights', shape=[inputs_len, 1], dtype=dtype)
      write_gate_bias = tf.get_variable('write_gate_bias', shape=[1], dtype=dtype)
      self.write_gate = tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, write_gate_weights, write_gate_bias), name='write_gate')

def one_plus(inputs, name=None):
  return tf.add(tf.exp(inputs), 1, name=name)

def assert_corner_cube(input_):
  assert_number = tf.Assert(tf.reduce_all(input_ != float('nan')), [input_], summarize=32, name='assert_number')
  assert_positive = tf.Assert(tf.reduce_all(input_ >= 0.0), [input_], summarize=32, name='assert_positive')
  assert_within_corner = tf.Assert(tf.reduce_all(tf.reduce_sum(input_, reduction_indices=1) <= 1.0001), [tf.reduce_sum(input_, reduction_indices=1)], summarize=32, name='assert_corner_cube')
  return with_dependencies([assert_number, assert_positive, assert_within_corner], input_)
