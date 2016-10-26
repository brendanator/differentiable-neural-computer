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
      read_weightings = tf.TensorShape([self.num_read_heads, self.memory_locations]),
      write_weighting = self.memory_locations)

  @property
  def output_size(self):
    # return tf.TensorShape([self.num_read_heads, self.memory_width])
    return self.num_read_heads * self.memory_width

  def __call__(self, inputs, state):
    with tf.variable_scope('MemoryNetwork'):
      # Extract inputs properties
      batch_size = inputs.get_shape().as_list()[0]
      dtype = inputs.dtype

      # Extract state
      memory, usage, precedence_weighting, temporal_linkage, read_weightings, write_weighting = state

      memory = tf.check_numerics(memory, 'memory')
      usage = tf.check_numerics(usage, 'usage')
      precedence_weighting = tf.check_numerics(precedence_weighting, 'precedence_weighting')
      temporal_linkage = tf.check_numerics(temporal_linkage, 'temporal_linkage')
      read_weightings = tf.check_numerics(read_weightings, 'read_weightings')
      write_weighting = tf.check_numerics(write_weighting, 'write_weighting')

      # Initiase variables
      read_heads = [ ReadHead(inputs, self.memory_locations, self.memory_width, i, dtype) for i in range(self.num_read_heads) ]
      write_head = WriteHead(inputs, self.memory_locations, self.memory_width, dtype)

      # Dynamic memory allocation
      with tf.variable_scope('memory_allocation'):
        memory_retention = tf.ones(shape=[batch_size, self.memory_locations], dtype=dtype)
        for index, read_head in enumerate(read_heads):
          memory_retention *= 1 - read_head.free_gate * read_weightings[:, index, :]

        # memory_retention = tf.Print(memory_retention, [memory_retention], 'memory_retention')

        new_usage = (usage + write_weighting - usage * write_weighting) * memory_retention

        sorted_usage, indices = tf.nn.top_k(new_usage, k=self.memory_locations, sorted=True)
        # sorted_usage = tf.Print(sorted_usage, [indices], 'sorted_usage', summarize=320)

        # This is extremely slow as it performs loads of operations
        allocation_weighting_list = []
        for batch in range(batch_size):
          scaling = 1
          weighting = [0] * self.memory_locations
          for i in range(self.memory_locations):
            location_usage = sorted_usage[batch, -i-1]
            weighting[i] = (1 - location_usage) * scaling
            scaling *= location_usage
          allocation_weighting_list.append(tf.gather(tf.pack(weighting, 0), indices[batch, :]))

        allocation_weighting = tf.pack(allocation_weighting_list, 0)
        # print(allocation_weighting)
        allocation_weighting = assert_corner_cube(allocation_weighting)
        # allocation_weighting = tf.Print(allocation_weighting, [allocation_weighting], 'allocation_weighting')
        # allocation_weighting = tf.zeros_like(allocation_weighting)

      # Write weighting
      with tf.variable_scope('write_weighting'):
        write_content_weighting = self.content_weighting(memory, write_head.write_key, write_head.write_strength)
        new_write_weighting = write_head.write_gate * (write_head.allocation_gate * allocation_weighting + (1 - write_head.allocation_gate) * write_content_weighting)
        new_write_weighting = assert_corner_cube(new_write_weighting)

      # Temporal memory linkage
      with tf.variable_scope('temporal_linkage'):
        new_temporal_linkage = tf.zeros(shape=[batch_size, self.memory_locations, self.memory_locations], dtype=dtype)
        write_weighting_matrix = tf.tile(tf.expand_dims(new_write_weighting, 2), [1, 1, self.memory_locations])
        precedence_weighting_matrix = tf.tile(tf.expand_dims(precedence_weighting, 2), [1, 1, self.memory_locations])
        new_temporal_linkage = ((1 - write_weighting_matrix - tf.transpose(write_weighting_matrix, [0, 2, 1])) * temporal_linkage
                                + write_weighting_matrix * tf.transpose(precedence_weighting_matrix, [0, 2, 1]))
        non_diagonal_ones = tf.matrix_set_diag(tf.ones([batch_size, self.memory_locations, self.memory_locations]), tf.zeros([batch_size, self.memory_locations]))
        new_temporal_linkage = new_temporal_linkage * non_diagonal_ones

        new_precedence_weighting = (1 - tf.reduce_sum(new_write_weighting, reduction_indices=1, keep_dims=True)) * precedence_weighting + new_write_weighting
        new_precedence_weighting = assert_corner_cube(new_precedence_weighting)
        for i in range(self.memory_locations):
          assert_corner_cube(new_temporal_linkage[i, :])
          assert_corner_cube(new_temporal_linkage[:, i])

      # Write memory
      with tf.variable_scope('write_memory'):
        new_memory = (
            memory * (1 - tf.batch_matmul(tf.expand_dims(new_write_weighting, 2), tf.expand_dims(write_head.erase_vector, 1)))
            + tf.batch_matmul(tf.expand_dims(new_write_weighting, 2), tf.expand_dims(write_head.write_vector, 1)))
        # new_memory = tf.Print(new_memory, [new_memory], 'memory', summarize=100)

      # Read memory
      with tf.variable_scope('read_memory'):
        new_read_weighting_list = []
        read_result_list = []
        for index, read_head in enumerate(read_heads):
          read_weighting = read_weightings[:, index, :]

          backward_weighting = tf.squeeze(tf.batch_matmul(tf.transpose(new_temporal_linkage, [0, 2, 1]), tf.expand_dims(read_weighting, 2)), [2])
          backward_weighting = assert_corner_cube(backward_weighting)
          read_content_weighting = self.content_weighting(new_memory, read_head.read_key, read_weighting)
          forward_weighting = tf.squeeze(tf.batch_matmul(new_temporal_linkage, tf.expand_dims(read_weighting, 2)), [2])
          forward_weighting = assert_corner_cube(forward_weighting)

          # read_head.read_mode = tf.Print(read_head.read_mode, [tf.reduce_sum(backward_weighting), tf.reduce_sum(read_content_weighting), tf.reduce_sum(forward_weighting)], 'read_head.read_mode')
          new_read_weighting = tf.batch_matmul(tf.expand_dims(read_head.read_mode, 1), tf.pack([backward_weighting, read_content_weighting, forward_weighting], 1))

          read_result = tf.squeeze(tf.batch_matmul(new_read_weighting, new_memory), [1])

          new_read_weighting_list.append(new_read_weighting)
          read_result_list.append(read_result)

        new_read_weighting = assert_corner_cube(new_read_weighting)

      # Return results
      output = tf.concat(1, read_result_list)
      # output = with_dependencies([tf.Assert(output != float('nan'), [output])], output)
      # output = tf.Print(output, [new_memory, new_usage, new_precedence_weighting, new_temporal_linkage, tf.concat(1, new_read_weighting_list, name='new_read_weighting'), new_write_weighting], 'output')
      new_state = MemoryNetworkState(new_memory, new_usage, new_precedence_weighting, new_temporal_linkage, tf.concat(1, new_read_weighting_list, name='new_read_weighting'), new_write_weighting)
      return output, new_state

  def content_weighting(self, memory, key, strength):
    with tf.variable_scope('content_weighting'):
      epsilon = 0.0001
      normalised_memory = memory / (tf.sqrt(tf.reduce_sum(tf.square(memory), 1, keep_dims=True)) + epsilon)
      normalised_key = key / (tf.sqrt(tf.reduce_sum(tf.square(key), 1, keep_dims=True)) + epsilon)

      content_weightings = tf.exp(tf.squeeze(tf.batch_matmul(normalised_memory, tf.expand_dims(normalised_key, 2))) * strength)
      total_content_weightings = tf.reduce_sum(content_weightings, reduction_indices=1, keep_dims=True)

      result = content_weightings / total_content_weightings
      assert_corner_cube(result)
      return result

class ReadHead(object):
  def __init__(self, inputs, memory_locations, memory_width, index, dtype):
    inputs_len = inputs.get_shape().as_list()[1]

    with tf.variable_scope('ReadHead%s' % index):
      read_key_weights = tf.get_variable('read_key_weights', shape=[inputs_len, memory_width], dtype=dtype)
      read_key_bias = tf.get_variable('read_key_bias', shape=[memory_width], dtype=dtype)
      self.read_key = tf.nn.xw_plus_b(inputs, read_key_weights, read_key_bias, name='read_key')

      read_strength_weights = tf.get_variable('read_strength_weights', shape=[inputs_len, 1], dtype=dtype)
      read_strength_bias = tf.get_variable('read_strength_bias', shape=[1], dtype=dtype)
      self.read_strength = one_plus(tf.nn.xw_plus_b(inputs, read_strength_weights, read_strength_bias), name='read_strength')

      free_gate_weights = tf.get_variable('free_gate_weights', shape=[inputs_len, 1], dtype=dtype)
      free_gate_bias = tf.get_variable('free_gate_bias', shape=[1], dtype=dtype)
      self.free_gate = tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, free_gate_weights, free_gate_bias), name='free_gate')

      read_mode_weights = tf.get_variable('read_mode_weights', shape=[inputs_len, 3], dtype=dtype)
      read_mode_bias = tf.get_variable('read_mode_bias', shape=[3], dtype=dtype)
      self.read_mode = tf.nn.softmax(tf.nn.xw_plus_b(inputs, read_mode_weights, read_mode_bias), name='read_mode')

class WriteHead(object):
  def __init__(self, inputs, memory_locations, memory_width, dtype):
    inputs_len = inputs.get_shape().as_list()[1]

    with tf.variable_scope('WriteHead'):
      write_key_weights = tf.get_variable('write_key_weights', shape=[inputs_len, memory_width], dtype=dtype)
      write_key_bias = tf.get_variable('write_key_bias', shape=[memory_width], dtype=dtype)
      self.write_key = tf.nn.xw_plus_b(inputs, write_key_weights, write_key_bias, name='write_key')

      write_strength_weights = tf.get_variable('write_strength_weights', shape=[inputs_len, 1], dtype=dtype)
      write_strength_bias = tf.get_variable('write_strength_bias', shape=[1], dtype=dtype)
      self.write_strength = one_plus(tf.nn.xw_plus_b(inputs, write_strength_weights, write_strength_bias), name='write_strength')

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

def one_plus(inputs, name):
  return tf.add(tf.exp(inputs), 1, name=name)

def assert_corner_cube(input_):
  assert_number = tf.Assert(tf.reduce_all(input_ != float('nan')), [input_], summarize=32, name='assert_number')
  assert_positive = tf.Assert(tf.reduce_all(input_ >= 0.0), [input_], summarize=32, name='assert_positive')
  assert_within_corner = tf.Assert(tf.reduce_all(tf.reduce_sum(input_, reduction_indices=1) <= 1.0001), [tf.reduce_sum(input_, reduction_indices=1)], summarize=32, name='assert_corner_cube')
  return with_dependencies([assert_number, assert_positive, assert_within_corner], input_)
