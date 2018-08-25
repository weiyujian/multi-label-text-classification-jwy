# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

def linear(args, output_size,bias, bias_start=0.0, initializer=None, scope=None, reuse=False):
    """
    Linear map: output[k] = sum_i(args[i] * W[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias temr or not.
        bias_start: inilizatized value of bias.
        initializer: weight initializer

        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    assert args is not None
    if not isinstance(args, (list, tuple)):
        args = [args]
    if initializer is None:
        initializer  = tf.random_uniform_initializer(minval=-0.1,maxval=0.1,seed=1234)
    
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "SimpleLinear", initializer=initializer, reuse=reuse):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable("Bias", [output_size], initializer=init_ops.constant_initializer(bias_start))
    return res + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, bias=True, scope=('highway_lin_{0}'.format(idx))))
            t = tf.sigmoid(linear(input_, size, bias=True, scope=('highway_gate_{0}'.format(idx))) + bias)
            output = t * g + (1. - t) * input_
            input_ = output
    return output
