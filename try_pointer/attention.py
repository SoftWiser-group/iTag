from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
import numpy as np

REMOVE_FACTOR = 10000


class Attention(Layer):
    def __init__(self, units, return_alphas=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True
        self.return_alphas = return_alphas
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Create a trainable weight variable for this layer.
        self.w_omega = self.add_weight(name='w_omega',
                                       shape=(input_dim, self.units),
                                       initializer='uniform',
                                       trainable=True)
        self.b_omega = self.add_weight(name='b_omega',
                                       shape=(self.units,),
                                       initializer='zeros',
                                       trainable=True)
        self.u_omega = self.add_weight(name='u_omega',
                                       shape=(self.units, 1),
                                       initializer='uniform',
                                       trainable=True)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, mask=None):
        input_dim = K.shape(x)[-1]
        v = K.tanh(K.dot(K.reshape(x, [-1, input_dim]), self.w_omega) + K.expand_dims(self.b_omega, 0))
        vu = K.dot(v, self.u_omega)
        vu = K.reshape(vu, K.shape(x)[:2])
        if mask is not None:
            m = K.cast(mask, K.floatx())
            m = m - 1
            m = m * REMOVE_FACTOR
            vu = vu + m
        alphas = K.softmax(vu)
        output = K.sum(x * K.expand_dims(alphas, -1), 1)
        if self.return_alphas:
            return [output] + [alphas]
        else:
            return output
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[2])
        if self.return_alphas:
            alphas_shape = [(input_shape[0], input_shape[1])]
            return [output_shape] + alphas_shape
        else:
            return output_shape


class CoAttention(Layer):
    def __init__(self, return_alphas=False, **kwargs):
        super(CoAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_alphas = return_alphas
    
    def build(self, input_shape):
        input_dim_t = input_shape[0][-1]
        input_dim_f = input_shape[1][-1]
        # Create a trainable weight variable for this layer.
        self.w_beta = self.add_weight(name='w_beta',
                                      shape=(input_dim_t, input_dim_f),
                                      initializer='uniform',
                                      trainable=True)
        super(CoAttention, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, mask=None):
        input_dim_t = K.shape(x[0])[-1]
        input_dim_f = K.shape(x[1])[-1]
        
        # remove padding values
        m_t = K.cast(mask[0], K.floatx())
        t = x[0] * K.expand_dims(m_t, -1)
        
        # remove padding values
        m_f = K.cast(mask[1], K.floatx())
        f = x[1] * K.expand_dims(m_f, -1)
        
        # compute affinity matrix
        C = K.dot(K.reshape(t, [-1, input_dim_t]), self.w_beta)
        C = K.reshape(C, [-1, K.shape(x[0])[1], input_dim_f])
        C = K.tanh(K.batch_dot(C, K.permute_dimensions(f, (0, 2, 1))))
        
        m_t = m_t - 1
        m_t = m_t * REMOVE_FACTOR
        alpha_t = K.max(C, axis=2) + m_t
        alpha_t = K.softmax(alpha_t)
        
        m_f = m_f - 1
        m_f = m_f * REMOVE_FACTOR
        alpha_f = K.max(C, axis=1) + m_f
        alpha_f = K.softmax(alpha_f)
        
        t_sum = K.sum(t * K.expand_dims(alpha_t, -1), 1)
        f_sum = K.sum(f * K.expand_dims(alpha_f, -1), 1)
        
        output = t_sum + f_sum
        
        return output
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], input_shape[0][2])
        if self.return_alphas:
            alphas_shape = [(input_shape[0][0], input_shape[0][1])]
            return [output_shape] + alphas_shape
        else:
            return output_shape


class TimeAttention(Layer):
    def __init__(self, units, return_alphas=False, **kwargs):
        super(TimeAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.return_alphas = return_alphas
    
    def build(self, input_shape):
        self.input_dim_en = input_shape[0][-1]
        self.input_en_times = input_shape[0][-2]
        self.input_dim_de = input_shape[1][-1]
        # Create a trainable weight variable for this layer.
        # w1
        self.w_en = self.add_weight(name='w_en', shape=(self.input_dim_en, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # w2
        self.w_de = self.add_weight(name='w_de', shape=(self.input_dim_de, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # nu
        self.nu = self.add_weight(name='nu', shape=(self.units, 1),
                                  initializer='glorot_uniform', trainable=True)
        super(TimeAttention, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, mask=None):
        en_seq = x[0]
        de_seq = x[1]
        input_de_times = K.shape(de_seq)[-2]

        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times*self.units))
        att_en = K.repeat(att_en, input_de_times)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times*input_de_times, self.units))

        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)

        co_m = att_en + att_de
        co_m = K.reshape(co_m, (-1, self.units))

        mu = K.dot(K.tanh(co_m), self.nu)

        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))
        alphas = K.softmax(mu)
        p_gen = K.sigmoid(mu)

        en_seq = K.reshape(en_seq, shape=(-1, self.input_en_times*self.input_dim_en))
        en_seq = K.repeat(en_seq, input_de_times)
        en_seq = K.reshape(en_seq, shape=(-1, input_de_times, self.input_en_times, self.input_dim_en))

        sum_en = K.sum(en_seq * K.expand_dims(alphas, -1), 2)

        # output = K.concatenate([de_seq, sum_en], -1)
        output = de_seq + sum_en
        if self.return_alphas:
            alphas = K.reshape(alphas, shape=(-1, input_de_times, self.input_en_times))
            p_gen = K.reshape(p_gen, shape=(-1, input_de_times, self.input_en_times))
            return [output] + [alphas] + [p_gen]
        else:
            return output
    
    def compute_mask(self, inputs, mask=None):
        return None
    

    def compute_output_shape(self, input_shape):
        # output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] + input_shape[1][-1])
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1])
        if self.return_alphas:
            alpha_shape = [(input_shape[1][0], input_shape[1][1], input_shape[0][1])]
            pgen_shape = [(input_shape[1][0], input_shape[1][1], input_shape[0][1])]
            return [output_shape] + alpha_shape + pgen_shape
        else:
            return output_shape


class TimeAttention_topical(Layer):
    """
    inputs: [encoder_outputs, decoder_outputs, topics]
    outputs: [decoder_outputs, decoder_alphas, decoder_pgen]
    
    input_shapes:[(batch_size, max_words, embedding_size), 
    (batch_size, max_label, embedding_size), (batch_size, num_topic)]
    output_shapes:[(batch_size, max_label, embedding_size), 
    (batch_size, max_label, max_words), (batch_size, max_label, max_words)]
    """
    def __init__(self, units, return_alphas=False, **kwargs):
        super(TimeAttention_topical, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.topic_num = 0
        self.return_alphas = return_alphas

    def build(self, input_shape):
        self.input_dim_en = input_shape[0][-1]
        self.input_en_times = input_shape[0][-2]
        self.input_dim_de = input_shape[1][-1]
        self.topic_num = input_shape[-1][-1]
        # Create a trainable weight variable for this layer.
        # w1
        self.w_en = self.add_weight(name='w_en', shape=(self.input_dim_en, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # w2
        self.w_de = self.add_weight(name='w_de', shape=(self.input_dim_de, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # nu
        self.nu = self.add_weight(name='nu', shape=(self.units, 1),
                                  initializer='glorot_uniform', trainable=True)
        self.wt = self.add_weight(name='wt',
                                 shape=(self.input_dim_en, self.topic_num),
                                 initializer='random_normal',
                                 trainable=True)
        super(TimeAttention_topical, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        en_seq = x[0]
        de_seq = x[1]
        topics = x[2]
        input_de_times = K.shape(de_seq)[-2]

        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.units))
        att_en = K.repeat(att_en, input_de_times)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * input_de_times, self.units))

        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)

        topics_w = K.dot(topics, K.transpose(self.wt))
        topics_w = K.repeat(topics_w, self.input_en_times * input_de_times)

        # print("Here:", att_de, att_en, topics_w)
        co_m = att_en + att_de + topics_w
        co_m = K.reshape(co_m, (-1, self.units))

        mu = K.dot(K.tanh(co_m), self.nu)

        mu = K.reshape(mu, shape=(-1, input_de_times, self.input_en_times))
        alphas = K.softmax(mu)
        p_gen = K.sigmoid(mu)

        en_seq = K.reshape(en_seq, shape=(-1, self.input_en_times * self.input_dim_en))
        en_seq = K.repeat(en_seq, input_de_times)
        en_seq = K.reshape(en_seq, shape=(-1, input_de_times, self.input_en_times, self.input_dim_en))

        sum_en = K.sum(en_seq * K.expand_dims(alphas, -1), 2)

        # output = K.concatenate([de_seq, sum_en], -1)
        output = de_seq + sum_en
        if self.return_alphas:
            alphas = K.reshape(alphas, shape=(-1, input_de_times, self.input_en_times))
            p_gen = K.reshape(p_gen, shape=(-1, input_de_times, self.input_en_times))
            return [output] + [alphas] + [p_gen]
        else:
            return output

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        # output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] + input_shape[1][-1])
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1])
        if self.return_alphas:
            alpha_shape = [(input_shape[1][0], input_shape[1][1], input_shape[0][1])]
            pgen_shape = [(input_shape[1][0], input_shape[1][1], input_shape[0][1])]
            return [output_shape] + alpha_shape + pgen_shape
        else:
            return output_shape


class MaskedTimeAttention(Layer):
    def __init__(self, units, return_alphas=False, **kwargs):
        print(return_alphas)
        super(MaskedTimeAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.input_dim_en = 0
        self.input_dim_de = 0
        self.input_en_times = 0
        self.input_de_times = 0
        self.return_alphas = return_alphas
    
    def build(self, input_shape):
        self.input_dim_en = input_shape[0][-1]
        self.input_en_times = input_shape[0][-2]
        self.input_dim_de = input_shape[1][-1]
        self.input_de_times = input_shape[1][-2]
        # Create a trainable weight variable for this layer.
        # w1
        self.w_en = self.add_weight(name='w_en', shape=(self.input_dim_en, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # w2
        self.w_de = self.add_weight(name='w_de', shape=(self.input_dim_de, self.units),
                                    initializer='glorot_uniform', trainable=True)
        # nu
        self.nu = self.add_weight(name='nu', shape=(self.units, 1),
                                  initializer='glorot_uniform', trainable=True)
        super(MaskedTimeAttention, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, mask=None):
        en_seq = x[0]
        de_seq = x[1]
        mask = x[2]

        if mask is not None:
            # remove padding values
            m_en = K.cast(mask, K.floatx())
            en_seq = en_seq * K.expand_dims(m_en, -1)
        
        # compute alphas
        att_en = K.dot(K.reshape(en_seq, (-1, self.input_dim_en)), self.w_en)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.units))
        att_en = K.repeat(att_en, self.input_de_times)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times * self.input_de_times, self.units))
        
        att_de = K.dot(K.reshape(de_seq, (-1, self.input_dim_de)), self.w_de)
        att_de = K.reshape(att_de, shape=(-1, self.input_de_times, self.units))
        att_de = K.repeat_elements(att_de, self.input_en_times, 1)
        
        co_m = att_en + att_de
        co_m = K.reshape(co_m, (-1, self.units))
        
        mu = K.dot(K.tanh(co_m), self.nu)

        if mask is not None:
            m_en = K.repeat_elements(m_en, self.input_de_times, 1)
            m_en = K.reshape(m_en, shape=(-1, 1))
            m_en = m_en - 1
            m_en = m_en * REMOVE_FACTOR
            mu = mu + m_en
        
        mu = K.reshape(mu, shape=(-1, self.input_de_times, self.input_en_times))
        alphas = K.softmax(mu)
        
        en_seq = K.reshape(en_seq, shape=(-1, self.input_en_times * self.input_dim_en))
        en_seq = K.repeat(en_seq, self.input_de_times)
        en_seq = K.reshape(en_seq, shape=(-1, self.input_de_times, self.input_en_times, self.input_dim_en))
        
        sum_en = K.sum(en_seq * K.expand_dims(alphas, -1), 2)
        
        output = K.concatenate([de_seq, sum_en], -1)

        if self.return_alphas:
            print(123)
            return [output] + [alphas]
        else:
            print(111)
            return output
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[1][0], input_shape[1][1], input_shape[0][-1] + input_shape[1][-1])
        return output_shape


class Masked(Layer):
    def __init__(self, **kwargs):
        super(Masked, self).__init__(**kwargs)
        self.supports_masking = True
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Masked, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, mask=None):
        output = x
        if mask is not None:
            # remove padding values
            m = K.cast(mask, K.floatx())
            output = x * K.expand_dims(m, -1)
        return output
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape


class MaskedGlobalAveragePooling1D(GlobalAveragePooling1D):
    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True

    def call(self, x, mask=None):
        mask = K.cast(mask, K.floatx())
        x = x * K.expand_dims(mask, -1)
        return K.sum(x, axis=1) / K.expand_dims(K.sum(mask, axis=1), -1)
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class MaskedGlobalMaxPooling1D(GlobalMaxPooling1D):
    def __init__(self, **kwargs):
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.supports_masking = True
    
    def call(self, x, mask=None):
        mask = K.cast(mask, K.floatx())
        r_mask = (mask - 1)*REMOVE_FACTOR
        x = x * K.expand_dims(mask, -1)
        x = x + K.expand_dims(r_mask, -1)
        return super(MaskedGlobalMaxPooling1D, self).call(x)
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


def test():
    MAX_LABELS = 3
    MAX_WORDS = 4
    EMBEDDING_DIM = 2
    ATTENTION_SIZE = 5
    seq_en = Input(shape=(MAX_WORDS, EMBEDDING_DIM))
    seq_de = Input(shape=(MAX_LABELS, EMBEDDING_DIM))
    output, alphas = TimeAttention(units=ATTENTION_SIZE, return_alphas=True)([seq_en, seq_de])
    model = Model([seq_en, seq_de], [output, alphas])
    en_data = np.random.rand(1, MAX_WORDS, EMBEDDING_DIM)
    de_data = np.random.rand(1, MAX_LABELS, EMBEDDING_DIM)
    res, alphas = model.predict([en_data, de_data])
    print(res)
    print(alphas.reshape((-1, MAX_LABELS, MAX_WORDS)))


if __name__=="__main__":
    input1 = Input(batch_shape=(10, 25, 50))
    input2 = Input(batch_shape=(10, 10, 50))
    input3 = Input(batch_shape=(10, 10))

    topic_h = TimeAttention_topical(50)([input1, input2, input3])
    print(topic_h)
