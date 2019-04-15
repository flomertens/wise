# -*- coding: utf-8 -*-
# This file is part of jsonpickle (https://github.com/jsonpickle/)
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009-2016 David Aguilar (davvid -at- gmail.com)
# All rights reserved.
#

from __future__ import absolute_import

import numpy as np

import ast
import jsonpickle
# import unicode from jsonpickle, depending on its version apparently function name is different.
try:
    from jsonpickle.compat import unicode
except ImportError:
    from jsonpickle.compat import ustr as unicode

__all__ = ['register_handlers', 'unregister_handlers']


class NumpyBaseHandler(jsonpickle.handlers.BaseHandler):

    def restore_dtype(self, data):
        dtype = data['dtype']
        if dtype.startswith(('{', '[')):
            return ast.literal_eval(dtype)
        return np.dtype(dtype)

    def flatten_dtype(self, dtype, data):

        if hasattr(dtype, 'tostring'):
            data['dtype'] = dtype.tostring()
        else:
            dtype = unicode(dtype)
            prefix = '(numpy.record, '
            if dtype.startswith(prefix):
                dtype = dtype[len(prefix):-1]
            data['dtype'] = dtype


class NumpyDTypeHandler(NumpyBaseHandler):

    def flatten(self, obj, data):
        self.flatten_dtype(obj, data)
        return data

    def restore(self, data):
        return self.restore_dtype(data)


class NumpyGenericHandler(NumpyBaseHandler):

    def flatten(self, obj, data):
        self.flatten_dtype(obj.dtype, data)
        data['value'] = self.context.flatten(obj.tolist(), reset=False)
        return data

    def restore(self, data):
        value = self.context.restore(data['value'], reset=False)
        return self.restore_dtype(data).type(value)


class NumpyNDArrayHandler(NumpyBaseHandler):

    def flatten(self, obj, data):
        self.flatten_dtype(obj.dtype, data)
        data['values'] = self.context.flatten(obj.tolist(), reset=False)
        return data

    def restore(self, data):
        dtype = self.restore_dtype(data)
        return np.array(self.context.restore(data['values'], reset=False),
                        dtype=dtype)


def register_handlers():
    jsonpickle.handlers.register(np.dtype, NumpyDTypeHandler)
    jsonpickle.handlers.register(np.generic, NumpyGenericHandler)
    jsonpickle.handlers.register(np.ndarray, NumpyNDArrayHandler)


def unregister_handlers():
    jsonpickle.handlers.unregister(np.dtype)
    jsonpickle.handlers.unregister(np.generic)
    jsonpickle.handlers.unregister(np.ndarray)
