import pathlib
import json
import logger
import tensorflow as tf
import numpy as np

import tools

class StudyBehaviour:
  def __init__(self, batch_size, vid_logdir, count_steps, fps=20):
    self._batch_size = batch_size
    self._vid_logdir = vid_logdir
    self._fps = fps
    self._count_steps = count_steps

    self._b_step = None
    self._b_data = {}
    self._b_size = 0

  def _load_metrics(self, metrics):
    for key, val in metrics:
      if key in self._b_data:
        self._b_data[key].append(val)
      else:
        self._b_data[key] = [val]
    self._b_size += 1

  def _write_logs(self, writer):
    with writer.as_default():  # Env might run in a different thread.
      tf.summary.experimental.set_step(self._b_step)
      for key, values in self._b_data.items():
        tf.summary.scalar(key + '/mean', np.mean(values))
        tf.summary.scalar(key + '/max', np.max(values))
        tf.summary.scalar(key + '/min', np.min(values))

    self._b_step = None
    self._b_size = 0
    self._b_data = {}

  def __call__(self, episode, config, datadir, writer, prefix):
    if self._b_step is None:
      self._b_step = self._count_steps(datadir, config)

    length = (len(episode['reward']) - 1) * config.action_repeat
    ret = episode['reward'].sum()
    print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
    metrics = [
        (f'{prefix}/return', float(ret)),
        (f'{prefix}/length', len(episode['reward']) - 1)]
    
    self._load_metrics(metrics)
    with (config.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps(dict([('step', self._b_step)] + metrics)) + '\n')

    dir = self._vid_logdir / 'step_{}'.format(self._b_step)
    dir.mkdir(parents=True, exist_ok=True)
    name = 'ep_{}_len_{}_ret_{}.mp4'.format(self._b_size, length, ret)
    logger.make_video(episode['image'], str(dir / name), self._fps)
    
    if self._b_size == self._batch_size:
      self._write_logs(writer)


class DataBuffer():

  def __init__(self, directory):
    self._ep_buffer = []
    self._ep_len = []
    self._directory = directory

    directory = pathlib.Path(self._directory).expanduser()
    cache = []
    for filename in directory.glob('*.npz'):
      if filename not in cache:
        try:
          with filename.open('rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
          print(f'Could not load episode: {e}')
          continue
        cache.append(filename)
        self._ep_buffer = episode
        self._ep_len = len(next(iter(episode.values())))

  def count_episodes(self):
    assert len(self._ep_buffer) == len(self._ep_len), 'Buffer corrupted!'
    return len(self._ep_buffer), sum(self._ep_len)

  def save_episode(self, episodes):
    for episode in episodes:
      self._ep_buffer.append(episode)
      self._ep_len.append(len(episode['reward']))
    tools.save_episodes(self.directory, episodes)

  def load_episodes(self, rescan, length=None, balance=False, seed=0):
    random = np.random.RandomState(seed)
    for index in random.choice(len(self._ep_buffer), rescan):
      episode = self._ep_buffer[index]
      if length:
        total = self._ep_len[index]
        available = total - length
        if available < 1:
          print(f'Skipped short episode of length {available}.')
          continue
        if balance:
          index = min(random.randint(0, total), available)
        else:
          index = int(random.randint(0, available))
        episode = {k: v[index: index + length] for k, v in episode.items()}
      yield episode
