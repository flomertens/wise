from features import *
from matcher import *
from scc import *
from wds import *
from project import *
from wiseutils import *
import tasks

# We need some extra bits to handle numpy array pickling correctly
import jsonpickle_numpy

import libwise

__version__ = '0.4.7'


def get_version():
    return '%s (libwise: %s)' % (__version__, libwise.get_version())


jsonpickle_numpy.register_handlers()
