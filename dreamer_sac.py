import argparse
import functools
import os
import pathlib
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

tf.get_logger().setLevel('ERROR')

sys.path.append(str(pathlib.Path(__file__).parent))

import models
import tools

from densedreamer import DenseDreamer, main


def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('./logs/dummy')
  config.seed = 0
  config.steps = 5e6
  config.eval_every = 1e4
  config.study_every = 1e5
  config.study_episodes = 5
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = False
  config.gpu_growth = True
  config.distributed = True
  config.precision = 16
  # Environment.
  config.task = 'dmc_walker_walk'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 2
  config.time_limit = 1000
  config.prefill = 5000
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  # Model.
  config.deter_size = 75
  config.stoch_size = 30
  config.num_units = 100
  config.embed_size = 50
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.dnn_depth = 3
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 1000
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.alpha_lr = 8e-5
  config.grad_clip = 100.0
  config.dataset_balance = False
  # sac parameters
  config.alpha = 'auto'
  config.polyak_factor = None
  config.t_update_freq = 1
  config.num_critics = 2
  # Behavior.
  config.discount = 0.99
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  return config


class SoftDenseDreamer(DenseDreamer):

  @tf.function()
  def train(self, data, log_images=False):
    if self._c.distributed:
      self._strategy.experimental_run_v2(self._train, args=(data, log_images))
    else:
      self._train(next(self._dataset), log_images)
    self._update_target()

  def _train(self, data, log_images):
    with tf.GradientTape() as model_tape:
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      obs_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      likes = tools.AttrDict()
      likes.obs = tf.reduce_mean(obs_pred.log_prob(data['obs']))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
      if self._c.pcont:
        pcont_pred = self._pcont(feat)
        pcont_target = self._c.discount * data['discount']
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
        likes.pcont *= self._c.pcont_scale
      prior_dist = self._dynamics.get_dist(prior)
      post_dist = self._dynamics.get_dist(post)
      div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
      div = tf.maximum(div, self._c.free_nats)
      model_loss = self._c.kl_scale * div - sum(likes.values())
      if self._c.distributed:
        model_loss /= float(self._strategy.num_replicas_in_sync)

    with tf.GradientTape() as actor_tape:
      imag_feat, imag_act = self._imagine_ahead(post)
      imag_st_act = tf.concat([imag_feat, imag_act['sample']], axis=-1)
      reward = self._reward(imag_feat[1:]).mode()
      qval_reward = reward - (self._alpha*self._c.discount) * imag_act['log_prob'][1:]
      if self._c.pcont:
        pcont = self._pcont(imag_feat[1:]).mean()
      else:
        pcont = self._c.discount * tf.ones_like(reward)
      qvalue = [self._qvalue[i](imag_st_act).mode() for i in range(self._c.num_critics)]
      qvalue = tf.reduce_min(qvalue, axis=0)
      returns = tools.lambda_return(
          qval_reward, qvalue[:-1], pcont,
          bootstrap=qvalue[-1], lambda_=self._c.disclam, axis=0)
      max_ent_return = returns - self._alpha * imag_act['log_prob'][:-1]
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-1]], 0), 0))
      actor_loss = -tf.reduce_mean(discount * max_ent_return)
      if self._c.distributed:
        actor_loss /= float(self._strategy.num_replicas_in_sync)

    if self._use_target:
      qvalue = [self._qvalue_t[i](imag_st_act).mode() for i in range(self._c.num_critics)]
      qvalue = tf.reduce_min(qvalue, axis=0)
      returns = tf.stop_gradient(tools.lambda_return(
                                  qval_reward, qvalue[:-1], pcont,
                                  bootstrap=qvalue[-1], lambda_=self._c.disclam, axis=0))
    qvalue_loss = []
    qvalue_tape = []
    for i in range(self._c.num_critics):
      with tf.GradientTape() as tape:
        qvalue_pred = self._qvalue[i](imag_st_act)[:-1]
        qvalue_loss.append(-tf.reduce_mean(discount * qvalue_pred.log_prob(tf.stop_gradient(returns))))
        if self._c.distributed:
          qvalue_loss[i] /= float(self._strategy.num_replicas_in_sync)
      qvalue_tape.append(tape)

    model_norm = self._model_opt(model_tape, model_loss)
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    qvalue_norm = [self._qvalue_opt[i](qvalue_tape[i], qvalue_loss[i]) 
                    for i in range(self._c.num_critics)]

    if self._c.alpha == 'auto':
      with tf.GradientTape() as alpha_tape:
        alpha_loss = -tf.reduce_mean(
                  self._alpha * tf.stop_gradient(imag_act['log_prob'][0] + self._target_entropy))
        if self._c.distributed:
            alpha_loss /= float(self._strategy.num_replicas_in_sync)
      alpha_norm = self._alpha_opt(alpha_tape, alpha_loss)
    else:
      alpha_norm, alpha_loss = (0, 0)

    if tf.distribute.get_replica_context().replica_id_in_sync_group == 0:
      if self._c.log_scalars:
        self._scalar_summaries(
            data, feat, prior_dist, post_dist, likes, div,
            model_loss, qvalue_loss, actor_loss, model_norm,
            qvalue_norm, actor_norm, alpha_loss, alpha_norm)

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    act = acts[self._c.dense_act]
    self._encode = models.DenseEncoder((self._c.embed_size,), self._c.dnn_depth,
                                       self._c.num_units)
    # self._encode = lambda x: x['obs']
    self._dynamics = models.RSSM(
        self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._decode = models.DenseDecoder((self._obsdim,), self._c.dnn_depth, 
                                       self._c.num_units)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    q_input = (self._c.stoch_size + self._c.deter_size + self._actdim,)
    self._qvalue = [models.DenseDecoderV2(q_input, (), 3, self._c.num_units, act=act)
                      for _ in range(self._c.num_critics)]
    self._use_target = (self._c.polyak_factor is not None or (self._c.t_update_freq is not None 
                        and self._c.t_update_freq > 1))
    if self._use_target:
      self._qvalue_t = [models.DenseDecoderV2(q_input, (), 3, self._c.num_units, act=act)
                          for _ in range(self._c.num_critics)]
      self._update_target = lambda: [self._copy_weights(self._qvalue_t[i], self._qvalue[i]) 
                                        for i in range(self._c.num_critics)]
      self._update_target()
    else:
      self._update_target = lambda: None
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)

    if self._c.alpha == 'auto':
      self._target_entropy = -self._actdim
      self._log_alpha = tf.Variable(0.0, dtype=self._float)
      self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)
      self._alpha_opt = tools.Adam('alpha', [], self._c.alpha_lr)
      self._alpha_opt._variables = [self._log_alpha]
    else:
      self._alpha = self._c.alpha

    model_modules = [self._encode, self._dynamics, self._decode, self._reward]
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)
    self._qvalue_opt = [Optimizer('qvalue_{}'.format(i+1), [self._qvalue[i]], self._c.value_lr)
                          for i in range(self._c.num_critics)]
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)

    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def _copy_weights(self, net_t, net):
    if self._c.polyak_factor is not None:
      def polyak_avg(target, policy):
        target.assign(self._c.polyak_factor * target + (1 - self._c.polyak_factor) * policy)
      tf.nest.map_structure(polyak_avg, net_t.variables, net.variables)
    elif self._c.t_update_freq is not None:
      net_t._last_t_update = (net_t._last_t_update + 1) if hasattr(net_t, '_last_t_update') else 0
      if net_t._last_t_update % self._c.t_update_freq == 0:
        net_t._last_t_update = 0
        tf.nest.map_structure(lambda x, y: x.assign(y), net_t.variables, net.variables)

  def _imagine_ahead(self, post):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in post.items()}
    action = {'sample': None, 'log_prob': None}

    def policy(feat):
      act_dist = self._actor(tf.stop_gradient(feat))
      action = {'sample': act_dist.sample()}
      action['log_prob'] = act_dist.log_prob(action['sample'])
      return action

    def scan_fn(prev, _):
      action = policy(self._dynamics.get_feat(prev[0]))
      return (self._dynamics.img_step(prev[0], action['sample']), action)

    states, actions = tools.static_scan(scan_fn,
                        tf.range(self._c.horizon), (start, action))
    imag_feat = self._dynamics.get_feat(states)
    last_act = policy(imag_feat[-1:])
    imag_action = tf.nest.map_structure(lambda x, y: tf.concat([x, y], axis=0),
                                        actions, last_act)
    imag_feat = tf.nest.map_structure(lambda x, y: tf.concat([x, y], axis=0),
                    tf.expand_dims(self._dynamics.get_feat(start), axis=0), imag_feat)
    return imag_feat, imag_action

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, qvalue_loss, actor_loss, model_norm, qvalue_norm,
      actor_norm, alpha_loss, alpha_norm):
    self._metrics['model_grad_norm'].update_state(model_norm)
    [self._metrics['qvalue_{}_grad_norm'.format(i+1)].update_state(qvalue_norm[i])
        for i in range(self._c.num_critics)]
    self._metrics['actor_grad_norm'].update_state(actor_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    [self._metrics['qvalue_{}_loss'.format(i+1)].update_state(qvalue_loss[i])
        for i in range(self._c.num_critics)]
    self._metrics['actor_loss'].update_state(actor_loss)
    self._metrics['action_ent'].update_state(tf.reduce_mean(self._actor(feat).entropy()))
    if self._c.alpha == 'auto':
      self._metrics['alpha'].update_state(tf.convert_to_tensor(self._alpha))
      self._metrics['alpha_loss'].update_state(alpha_loss)
      self._metrics['alpha_norm'].update_state(alpha_norm)


if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(SoftDenseDreamer, parser.parse_args())
