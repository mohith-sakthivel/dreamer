import pathlib
import json
import cv2
import tensorflow as tf
import numpy as np


import os.path as osp

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
    make_video(episode['image'], str(dir / name), self._fps)
    
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


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) 
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v) 
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def save_config(config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        print('Saving config...')
        with open(osp.join(config.logdir, "config.json"), 'w') as out:
            out.write(output)

def make_video(img_list, video_dir='videos/vid.mp4', fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    img_list = [cv2.resize(img, (256, 256)) for img in img_list]
    height, width = img_list[0].shape[:-1]
    out = cv2.VideoWriter(video_dir, fourcc, fps, (width, height))
    for image in img_list:  
        out.write(image)
    out.release()


class IntervalCheck():
    def __init__(self, interval=None):
        self._interval = interval
        self._last = 0

    def __call__(self, curr_counter):
        if self._interval is not None and  (curr_counter // self._interval) > self._last:
            self._last = curr_counter // self._interval
            return True
        return False
