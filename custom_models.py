import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow.python.eager.context import context
from tensorflow.python.keras.backend import dtype, zeros
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools


class ContextEncoder(tools.Module):

    def __init__(self, latent_dim, encoder_dim, 
                 hidden=256, pre_layers=0, post_layers=0,
                 state_embed=32, act_embed=16, rew_embed=16,
                 act=tf.nn.elu, dist='gaussian', min_std=1e-4):
        super().__init__()
        self._latent_dim = latent_dim
        self._encoder_dim = encoder_dim
        self._hidden = hidden
        self._pre_layers = pre_layers
        self._post_layers = post_layers
        self._embed_dim = {}
        self._embed_dim ['state']= state_embed
        self._embed_dim['act'] = act_embed
        self._embed_dim['rew'] = rew_embed
        self._act = act
        self._dist = dist
        self._min_std = min_std
        # TODO: TEST different encoder choices GRU, Attention+TC
        self._cell = tfkl.GRU(self._encoder_dim, return_sequences=True, return_state=True)

    def _get_embed(self, item):
        return (self.get(f'{item}_embed', tfkl.Dense, self._embed_dim[item], self._act)
                        if self._embed_dim[item] is not None
                        else lambda x: x)

    def get_initial_state(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        ctor = lambda *args, **kwargs: tf.Variable(
            tf.random_normal_initializer(mean=0, stddev=0.1)(shape=(self._encoder_dim,), dtype=dtype), *args, **kwargs)
        init_hid_state =  self.get(f'init_hidden_state', ctor)
        init_hid_state = tf.expand_dims(init_hid_state, axis=0)
        return tf.tile(init_hid_state, (batch_size, 1))

    def __call__(self, state, action, reward, hid_state, training=True, detach_every=None):
        state_embed = self._get_embed('state')(state)
        act_embed = self._get_embed('act')(action)
        rew_embed = self._get_embed('rew')(tf.expand_dims(reward, -1))
        x = tf.concat([state_embed, act_embed, rew_embed], axis=-1)
        detach_every = state.shape[1] if detach_every is None else detach_every

        for index in range(self._pre_layers):
            x = self.get(f'pre_h{index}', tfkl.Dense, self._units, self._act)(x)

        if training:
          x, hid_state = self._cell(x, hid_state)
        else:
          x = tf.expand_dims(x, axis=1)
          x, hid_state = self._cell(x, hid_state)
          x = x[:, 0]

        for index in range(self._post_layers):
            x = self.get(f'post_h{index}', tfkl.Dense, self._units, self._act)(x)
        
        if self._dist == 'gaussian':
            x = self.get(f'hout', tfkl.Dense, 2*self._latent_dim)(x)
            mean, std_t = tf.split(x, 2, -1)
            std = tf.nn.softplus(std_t) + self._min_std
            dist = tfd.Independent(tfd.Normal(mean, std), 1)
            out = {'mean': mean, 'std': std}
        else:
            raise NotImplementedError
        out['sample'] = dist.sample()
        return out, hid_state
        

class CB_RSSM(tools.Module):

  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, context, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2])
    action = tf.transpose(action, [1, 0, 2])
    context = tf.transpose(context, [1, 0, 2])
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed, context), (state, state))
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, context):
    prior = self.img_step(prev_state, prev_action, context)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, context):
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    # x = tf.concat([context, x], axis=-1) #TODO: Integrate forward dynamics prediction
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior    