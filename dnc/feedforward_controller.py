import tensorflow as tf

class FeedforwardController():
  @property
  def output_size(self):
    """Integer or 1-D TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def feedforward(self, inputs):
    """Run this controller on inputs.

    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.

    Returns:
      A `2-D` tensor with shape `[batch_size x self.output_size]`.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    return 0

  def __call__(self, inputs, state):
    return self.feedforward(inputs), state

class SimpleFeedforwardController(FeedforwardController):
  def __init__(self, num_units, num_layers, activation=tf.nn.sigmoid):
    self.num_units = num_units
    self.num_layers = num_layers
    self.activation = activation

  @property
  def output_size(self):
    return self.num_units

  def feedforward(self, inputs):
    inputs_size = inputs.get_shape().as_list()[1]
    dtype = inputs.dtype

    with tf.variable_scope('SimpleFeedforwardController'):
      output = self.layer(0, inputs, inputs_size, self.num_units, dtype)
      for i in range(1, self.num_layers):
        output = self.layer(i, output, self.num_units, self.num_units, dtype)

    return output

  def layer(self, layer, inputs, inputs_size, output_size, dtype):
    with tf.variable_scope('layer-%d' % layer):
      weight = tf.get_variable('weight', shape=[inputs_size, output_size], dtype=dtype)
      bias = tf.get_variable('bias', shape=[output_size], dtype=dtype, initializer=tf.constant_initializer(0.1, dtype=dtype))
      return self.activation(tf.matmul(inputs, weight) + bias)
