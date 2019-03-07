from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import tf_logging as logging

def lp_regularizer(scale, p=2, scope=None):
  """Returns a function that can be used to apply Lp regularization to weights.
  Args:
    scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
    scope: An optional scope name.
    p: index
  Returns:
    A function with signature `lp(weights)` that applies Lp regularization.
  Raises:
    ValueError: If scale is negative or if scale is not a float.
  """
  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % (scale,))
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def lp(weights):
    """Applies l2 regularization to weights."""
    with ops.name_scope(scope, 'lp_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      reg_loss = standard_ops.reduce_sum(math_ops.pow(math_ops.abs(weigths), p))
      return standard_ops.multiply(my_scale, reg_loss, name=name)

return lp
