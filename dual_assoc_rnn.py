import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
import random
import functools


class AssociativeGRUCell(RNNCell):
    """
    An RNNCell that keeps a complex redundant associative memory arrays which can be read from or written to.

    It is recommended to use this cell wrapped in a SelfControllerCell. If used in this setup you can retrieve
    the redundant memory arrays by slicing the final state of the SelfControllerCell from [0,0]
    to [-1,num_units*num_copies]. This can be used for example to initialize the state of the DualAssociativeGRUCell.
    """

    def __init__(self, num_units, num_copies=1, num_read_keys=0, read_only=False, rng=None):
        """
        :param num_units: number of hidden units
        :param num_copies: number of redundant copies (the more the preciser the retrieval but slower)
        :param num_read_keys: number of keys used to read from the associative memory, we always read from the
        write location so actual reads are num_read_keys + 1
        :param read_only: if read only, no updates are performed on the memory array
        :param rng: optional,
        """
        if rng is None:
            rng = random.Random(123)
        self._num_units = num_units
        self._num_copies = num_copies
        self._num_read_keys = num_read_keys
        self._read_only = read_only
        self._permutations = [list(range(0, int(num_units/2))) for _ in range(self._num_copies-1)]
        for perm in self._permutations:
            rng.shuffle(perm)

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units * self._num_copies

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "AssociativeGRUCell"):
            if self._num_copies > 1:
                with vs.variable_scope("Permutations"):
                    perms = functools.reduce(lambda x, y: x+y, self._permutations)
                    perms = tf.constant(perms)

            old_ss = state
            c_ss = complexify(old_ss)
            with vs.variable_scope("Keys"):
                key = bound(complexify(tf.contrib.layers.fully_connected(inputs,
                                                                         (1+self._num_read_keys)*self._num_units,
                                                                         activation_fn=None, biases_initializer=None)))
                k = [_comp_real(key), _comp_imag(key)]
                if self._num_copies > 1:
                    if self._num_read_keys > 0:
                        k = tf.transpose(tf.concat(0, tf.split(1, 1+self._num_read_keys, k[0]) + tf.split(1, 1+self._num_read_keys, k[1])), [1, 0])
                    else:
                        k = tf.transpose(tf.concat(0, k), [1, 0])
                    k = tf.concat(0, [k, tf.gather(k, perms)])
                    k = tf.split(0, 2*(1+self._num_read_keys), tf.transpose(k, [1, 0]))

                w_k_real = k[0]
                w_k_imag = k[1+self._num_read_keys]
                r_k_real = k[1:1+self._num_read_keys]
                r_k_imag = k[2+self._num_read_keys:]
                r_keys = list(zip(r_k_real, r_k_imag))
                w_key = (w_k_real, w_k_imag)
                conj_w_key = _comp_conj(w_key)

            with vs.variable_scope("Read"):
                h = uncomplexify(self._read(conj_w_key, c_ss), "retrieved")
                r_hs = []
                for i, k in enumerate(r_keys):
                    r_hs.append(uncomplexify(self._read(k, c_ss), "read_%d" % i))

            if not self._read_only:
                with vs.variable_scope("Gates"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    gates = tf.contrib.layers.fully_connected(tf.concat(1, r_hs + [inputs, h]), 2 * self._num_units,
                                                              activation_fn=tf.sigmoid,
                                                              biases_initializer=tf.constant_initializer(1.0))
                    r, u = array_ops.split(1, 2, gates)
                with vs.variable_scope("Candidate"):
                    c = tf.contrib.layers.fully_connected(tf.concat(1, r_hs + [inputs, r * h]), self._num_units,
                                                          activation_fn=tf.tanh)

                to_add = u * (c - h)
                to_add_r, to_add_i = tf.split(1, 2, to_add)
                c_to_add = (tf.tile(to_add_r, [1, self._num_copies]), tf.tile(to_add_i, [1, self._num_copies]))
                new_ss = old_ss + uncomplexify(_comp_mul(w_key, c_to_add))
                new_h = tf.add(h, to_add, "out")
            else:
                new_h = h
                new_ss = old_ss

        return new_h, new_ss

    def _read(self, keys, redundant_states):
        read = _comp_mul(keys, redundant_states)
        if self._num_copies > 1:
            xs_real = tf.split(1, self._num_copies, _comp_real(read))
            xs_imag = tf.split(1, self._num_copies, _comp_imag(read))
            read = (tf.add_n(xs_real)/self._num_copies, tf.add_n(xs_imag)/self._num_copies)
        return read


class DualAssociativeGRUCell(AssociativeGRUCell):
    """
    To make use of the DualAssociativeGRUCell, the initial state should be composed of the an arbitrary associative
    memory with its redundant copies (e.g., [B, num_units * num_copies]-0-tensor) concatenated with the
    associative memories to read from with their respective copies
    (num_read_mems x [B, num_units * num_copies]-tensors that can be retrieved from final state of another
    AssociativeGRUCell.

    It is recommended to use this cell wrapped in a SelfControllerCell.
    """

    def __init__(self, num_units, num_read_mems=1, num_copies=1, share_key=False,
                 num_read_keys=0, read_only=False, rng=None):
        """
        :param num_units: number of hidden units
        :param num_read_mems: number of memories to read from
        :param num_copies: number of redundant copies (the more the preciser the retrieval but slower)
        :param num_read_keys: number of keys used to read from the associative memory, we always read from the
        write location so actual reads are num_read_keys + 1
        :param read_only: if read only, no updates are performed on the memory array
        :param rng: optional, must be equal to the rng used to create read-only memories used for dual access
        """
        self._num_read_mems = num_read_mems
        self._share_key = share_key
        AssociativeGRUCell.__init__(self, num_units, num_copies, num_read_keys, read_only, rng)

    @property
    def state_size(self):
        return self._num_units * self._num_copies * (1+self._num_read_mems)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "AssociativeGRUCell"):
            if self._num_copies > 1:
                with vs.variable_scope("Permutations"):
                    perms = functools.reduce(lambda x, y: x+y, self._permutations)
                    perms = tf.constant(perms)

            old_ss = tf.slice(state, [0, 0], [-1, self._num_units * self._num_copies])
            dual_mem = tf.slice(state, [0, self._num_units * self._num_copies], [-1, -1])
            read_mems = tf.split(1, self._num_read_mems, dual_mem)
            c_ss = complexify(old_ss)
            with vs.variable_scope("Keys"):
                num_keys = (1+self._num_read_keys)
                if not self._share_key:
                    num_keys += self._num_read_mems

                key1 = tf.contrib.layers.fully_connected(inputs, (1+self._num_read_keys)*self._num_units,
                                                         activation_fn=None, biases_initializer=None)
                with vs.variable_scope("Dual"):
                    vs.get_variable_scope()._reuse = \
                        any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())
                    key2 = tf.contrib.layers.fully_connected(inputs, self._num_read_mems*self._num_units,
                                                             activation_fn=None, biases_initializer=None)
                key = bound(complexify(tf.concat(1, [key1, key2])))

                k = [_comp_real(key), _comp_imag(key)]
                if self._num_copies > 1:
                    if num_keys > 0:
                        k = tf.transpose(tf.concat(0, tf.split(1, num_keys, k[0]) + tf.split(1, num_keys, k[1])), [1, 0])
                    else:
                        k = tf.transpose(tf.concat(0, k), [1, 0])
                    k = tf.concat(0, [k, tf.gather(k, perms)])
                    k = tf.split(0, 2*num_keys, tf.transpose(k, [1, 0]))
                w_k_real = k[0]
                w_k_imag = k[num_keys]
                r_k_real = k[1:num_keys]
                r_k_imag = k[num_keys+1:]
                r_keys = list(zip(r_k_real, r_k_imag))
                w_key = (w_k_real, w_k_imag)
                conj_w_key = _comp_conj(w_key)
                if self._share_key:
                    dual_keys = [conj_w_key for _ in range(self._num_read_mems)]
                else:
                    dual_keys = r_keys[self._num_read_keys:]
                    r_keys = r_keys[:self._num_read_keys]

            with vs.variable_scope("Read"):
                h = uncomplexify(self._read(conj_w_key, c_ss), "retrieved")
                r_hs = []
                for i, k in enumerate(r_keys):
                    r_hs.append(uncomplexify(self._read(k, c_ss), "read_%d" % i))

            with vs.variable_scope("Read_Given"):
                if self._num_read_mems > 1:
                    h2_s = [uncomplexify(self._read(dual_key, complexify(read_mem))) for read_mem, dual_key in zip(read_mems, dual_keys)]
                    h2 = tf.reshape(tf.concat(1, h2_s, name="retrieved"), [-1, self._num_read_mems*self._num_units])
                else:
                    h2 = uncomplexify(self._read(dual_keys[0], complexify(read_mems[0])), "retrieved")

            with vs.variable_scope("Gates"):
                gates = tf.contrib.layers.fully_connected(tf.concat(1, r_hs + [inputs, h]), 2 * self._num_units,
                                                          activation_fn=None,
                                                          biases_initializer=tf.constant_initializer(1.0))
            with vs.variable_scope("DualGates"):
                # HACK to make it possible to share all non-dual variables between
                # AssociativeGRUCell and DualAssociativeGRUCell
                vs.get_variable_scope()._reuse = \
                    any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())
                gates = sigmoid(gates + tf.contrib.layers.fully_connected(h2, 2 * self._num_units,
                                                                          activation_fn=None,
                                                                          biases_initializer=None))
            r, u = tf.split(1, 2, gates)

            with vs.variable_scope("Candidate"):
                c = tf.contrib.layers.fully_connected(tf.concat(1, r_hs + [inputs, r * h]), self._num_units,
                                                      activation_fn=None)

            with vs.variable_scope("DualCandidate"):
                # HACK to make it possible to share all non-dual variables between
                # AssociativeGRUCell and DualAssociativeGRUCell
                vs.get_variable_scope()._reuse = \
                    any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())  # HACK
                dual_c = tf.contrib.layers.fully_connected(h2, self._num_units, activation_fn=None,
                                                           biases_initializer=None)
                c = tanh(c + dual_c)

            to_add = u * (c - h)
            to_add_r, to_add_i = tf.split(1, 2, to_add)
            c_to_add = (tf.tile(to_add_r, [1, self._num_copies]), tf.tile(to_add_i, [1, self._num_copies]))
            new_ss = old_ss + uncomplexify(_comp_mul(w_key, c_to_add))
            new_h = tf.add(h, to_add, "out")

        return new_h, tf.concat(1, [new_ss, dual_mem])


class ControllerWrapper(RNNCell):
    """
    A ControllerWrapper wraps a controller RNNCell and controlled RNNCell. The controller cell pre-processes
    the current input and latest output of the controlled cell and feeds its output to the controlled cell.
    Like this Attention can be realized using the AttentionCell as controlled cell and another RNNCell as
    controller cell. The output of this wrapper is the concatenation of the controller cell output and the
    controlled cell output. This output can also be projected via a output projection function.
    """

    def __init__(self, controller_cell, cell, output_proj=None, out_size=None):
        """

        :param controller_cell: controller that controls the input to the cell
        :param cell: controlled cell, typically some form of memory
        :param output_proj: optional, function applied to the output of this cell to form new output
        :param out_size: optional, size of the respective output when using output_proj
        :return:
        """
        self._cell = cell
        self._controller_cell = controller_cell
        self._output_proj = output_proj
        self._out_size = out_size

    @property
    def output_size(self):
        if self._out_size is None:
            return self._controller_cell.output_size + self._cell.output_size
        else:
            return self._out_size

    def __call__(self, inputs, state, scope=None):
        ctr_state = tf.slice(state, [0, 0], [-1, self._controller_cell.state_size])
        inner_state = None
        if self._cell.state_size > 0:
            inner_state = tf.slice(state, [0, self._controller_cell.state_size], [-1, self._cell.state_size])
        inner_out = tf.slice(state, [0, self._controller_cell.state_size+self._cell.state_size], [-1,-1])
        inputs = tf.concat(1, [inputs, inner_out])
        ctr_out, ctr_state = self._controller_cell(inputs, ctr_state)
        inner_out, inner_state = self._cell(ctr_out, inner_state)
        out = tf.concat(1, [ctr_out, inner_out])
        if self._output_proj is not None:
            with tf.variable_scope("Output_Projection"):
                out = self._output_proj(out, self.output_size)
        if self._cell.state_size > 0:
            return out, tf.concat(1, [ctr_state, inner_state, inner_out])
        else:
            return out, tf.concat(1, [ctr_state, inner_out])

    def zero_state(self, batch_size, dtype):
        if self._cell.state_size > 0:
            return tf.concat(1, [self._controller_cell.zero_state(batch_size, dtype), self._cell.zero_state(batch_size, dtype),
                             tf.zeros([batch_size,self._cell.output_size]),tf.float32])
        else:
            return tf.concat(1, [self._controller_cell.zero_state(batch_size, dtype), tf.zeros([batch_size,self._cell.output_size]),tf.float32])

    @property
    def state_size(self):
        return self._controller_cell.state_size + self._cell.state_size + self._cell.output_size


class SelfControllerWrapper(RNNCell):
    """
    A SelfControllerWrapper wraps a RNNCell that controls itself. It basically concatenates its last output
    with the current input and feeds it as input to the wrapped cell. It is useful for example in combination
    with the (Dual)AssociativeGRUCell which doesn't reuse its previous output recurrently by itself. Other
    RNNCells like the GRU or LSTM already reuse their last outputs so this wrapper would not help there.
    This output can also be projected via a output projection function.
    """

    def __init__(self, cell, output_proj=None, out_size=None):
        """
        :param cell: the self-controlled RNNCell
        :param output_proj: optional, function applied to the output of this cell to form new output
        :param out_size: optional, size of the respective output when using output_proj
        :return:
        """
        self._cell = cell
        self._output_proj = output_proj
        self._out_size = out_size

    @property
    def output_size(self):
        if self._out_size is None:
            return self._cell.output_size
        else:
            return self._out_size

    def __call__(self, inputs, state, scope=None):
        prev_state = None
        if self._cell.state_size > 0:
            prev_state = tf.slice(state, [0, 0], [-1, self._cell.state_size])
        prev_out = tf.slice(state, [0, self._cell.state_size], [-1, self._cell.output_size])
        inputs = tf.concat(1, [inputs, prev_out])
        new_out, prev_state = self._cell(inputs, prev_state)
        out = new_out
        if self._output_proj is not None:
            with tf.variable_scope("Output_Projection"):
                out = self._output_proj(out, self.output_size)
        if self._cell.state_size > 0:
            return out, tf.concat(1, [prev_state, new_out])
        else:
            return out, new_out

    def zero_state(self, batch_size, dtype):
        if self._cell.state_size > 0:
            return tf.concat(1, [self._cell.zero_state(batch_size, dtype), tf.zeros(tf.pack([batch_size,self._cell.output_size]),tf.float32)])
        else:
            return tf.zeros(tf.pack([batch_size,self._cell.output_size]),tf.float32)

    @property
    def state_size(self):
        return self._cell.state_size + self._cell.output_size


class AttentionCell(RNNCell):
    """
    This RNNCell only makes sense in conjunction with ControllerWrapper
    """

    def __init__(self, attention_states, attention_length, num_heads=1):
        """
        :param attention_states: [B, L, S]-tensor, B-batch_size, L-attention-length, S-attention-size
        :param attention_length: [B]-tensor, with batch-specific sequence-lengths to attend over
        :param num_heads: number of attention read heads
        :return:
        """
        self._attention_states = attention_states
        self._attention_length = attention_length
        self._num_heads = num_heads
        self._hidden_features = None
        self._num_units = self._attention_states.get_shape()[2].value

    @property
    def output_size(self):
        return self._attention_states.get_shape()[2].value * self._num_heads

    def __call__(self, inputs, state, scope=None):
        if self._hidden_features is None:
            self._attn_length = math_ops.reduce_max(self._attention_length)
            attention_states = tf.slice(self._attention_states, [0,0,0], tf.pack([-1, self._attn_length, -1]))
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            self._hidden = array_ops.reshape(attention_states, tf.pack([-1, self._attn_length, 1, self._num_units]))
            hidden_features = []

            for a in range(self._num_heads):
                k = tf.get_variable("AttnW_%d" % a, [1, 1, self._num_units, self._num_units])
                hidden_features.append(nn_ops.conv2d(self._hidden, k, [1, 1, 1, 1], "SAME"))
            self._hidden_features = hidden_features


        ds = []  # Results of attention reads will be stored here.

        batch_size = tf.shape(inputs)[0]
        mask = tf.tile(tf.reshape(tf.lin_space(1.0, tf.cast(self._attn_length, tf.float32), self._attn_length), [1, -1]),
                       tf.pack([batch_size, 1]))
        batch_size_scale = batch_size // tf.shape(self._attention_length)[0] # used in decoding
        lengths = tf.tile(tf.expand_dims(tf.cast(self._attention_length, tf.float32), 1),
                          tf.pack([batch_size_scale, self._attn_length]))

        mask = tf.cast(tf.greater(mask, lengths), tf.float32) * -1000.0
        # some parts are copied from tensorflow attention code-base
        for a in range(self._num_heads):
            with tf.variable_scope("Attention_%d" % a):
                y = tf.contrib.layers.fully_connected(inputs, self._num_units, activation_fn=None)
                y = array_ops.reshape(y, [-1, 1, 1, self._num_units])
                # Attention mask is a softmax of v^T * tanh(...).
                v = tf.get_variable("AttnV_%d" % a, [self._num_units])
                hf = tf.cond(tf.equal(batch_size_scale,1),
                             lambda: self._hidden_features[a],
                             lambda: tf.tile(self._hidden_features[a], tf.pack([batch_size_scale,1,1,1])))
                s = math_ops.reduce_sum(v * math_ops.tanh(hf + y), [2, 3])

                a = nn_ops.softmax(s + mask)
                # Now calculate the attention-weighted vector d.
                d = math_ops.reduce_sum(
                    array_ops.reshape(a, tf.pack([-1, self._attn_length, 1, 1])) * self._hidden,
                    [1, 2])
                ds.append(array_ops.reshape(d, [-1, self._num_units]))

        return tf.concat(1, ds), None

    def zero_state(self, batch_size, dtype):
        return super().zero_state(batch_size, dtype)

    @property
    def state_size(self):
        return 0


### The following helper functions for complex array are much faster than using native tensorflow complex datastructure.

def complexify(v):
    v_r, v_i = tf.split(1, 2, v)
    return v_r, v_i


def uncomplexify(v, name=None):
    return tf.concat(1, v, name)


def bound(v):
    im_v = _comp_imag(v)
    re_v = _comp_real(v)
    v_modulus = tf.maximum(1.0, tf.sqrt(im_v * im_v + re_v * re_v))
    return re_v / v_modulus, im_v / v_modulus


def _comp_conj(x):
    return _comp_real(x), -_comp_imag(x)


def _comp_add(x, y):
    return _comp_real(x)+_comp_real(y), _comp_imag(x)+_comp_imag(y)


def _comp_add_n(xs):
    xs_real = [_comp_real(x) for x in xs]
    xs_imag = [_comp_imag(x) for x in xs]
    return tf.add_n(xs_real), tf.add_n(xs_imag)


def _comp_mul(x, y):
    return _comp_real(x) * _comp_real(y) - _comp_imag(x) * _comp_imag(y), \
           _comp_real(x) * _comp_imag(y) + _comp_imag(x) * _comp_real(y)


def _comp_real(x):
    return x[0]


def _comp_imag(x):
    return x[1]