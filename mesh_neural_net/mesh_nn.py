import tensorflow as tf


class MeshNeuralNetwork(tf.keras.layers.Layer):
    def __init__(self, num_hidden, num_outputs, num_ticks=3, use_bias=True,
                 activation=None,kernel_initializer=tf.initializers.RandomUniform(minval=0,maxval=1)):
        super(MeshNeuralNetwork, self).__init__()
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_ticks = num_ticks
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        self._hidden_out_size = self.num_hidden + self.num_outputs
        input_size = int(input_shape[-1]) + int(self.use_bias)
        self._out_idx = self.num_hidden
        self.kernel = self.add_weight("kernel", shape=[input_size + self._hidden_out_size,self._hidden_out_size],
                                      dtype=self.dtype, initializer=self.kernel_initializer)

    @tf.function
    def call(self, input):
        sh = tf.shape(input)
        state_shape = tf.concat([sh[:-1],tf.expand_dims(tf.constant(self._hidden_out_size,dtype=tf.int32),axis=0)],axis=0)
        state = tf.zeros(state_shape, dtype=self.dtype)
        out_begin_idxs = tf.concat([tf.zeros(len(sh)-1,dtype=tf.int32), tf.expand_dims(tf.constant(self.num_hidden,dtype=tf.int32),axis=0)],axis=0)
        end_idxs = -tf.ones([len(sh)],dtype=tf.int32)

        if self.use_bias:
            ones = tf.ones(tf.concat([sh[:-1],tf.expand_dims(tf.constant(1,dtype=tf.int32),axis=0)],axis=0), dtype=self.dtype)

        for i in range(self.num_ticks):
            if self.use_bias:
                state = tf.concat((input, ones, state), axis=-1)
            else:
                state = tf.concat((input, state), axis=-1)
            state = tf.matmul(state, self.kernel)
            if self.activation is not None:
                state = self.activation(state)
        out = tf.slice(state, out_begin_idxs, end_idxs)
        return out
