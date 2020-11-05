import json

import os.path as osp

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