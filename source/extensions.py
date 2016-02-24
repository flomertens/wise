'''From  http://stackoverflow.com/questions/20569011/python-sphinx-autosummary-automated-listing-of-member-functions '''

import re
import inspect

from sphinx.ext.autosummary import Autosummary
from sphinx.ext.autosummary import get_documenter
from docutils.parsers.rst import directives
from sphinx.util.inspect import safe_getattr

class AutoSummaryFromList(Autosummary):

    option_spec = {
        'list_name': directives.unchanged,
    }

    required_arguments = 1

    # @staticmethod
    # def get_members(obj, typ, include_public=None):
    #     if not include_public:
    #         include_public = []
    #     items = []
    #     for name in dir(obj):
    #         try:
    #             documenter = get_documenter(safe_getattr(obj, name), obj)
    #         except AttributeError:
    #             continue
    #         if documenter.objtype == typ:
    #             items.append(name)
    #     public = [x for x in items if x in include_public or not x.startswith('_')]
    #     return public, items

    def run(self):
        module_name = self.arguments[0]

        m = __import__(module_name, globals(), locals())
        list_module_name, list_name = self.options['list'].rsplit('.', 1)
        list_module = __import__(list_module_name, globals(), locals(), [list_name])
        allowed_list = getattr(list_module, list_name)

        self.content = ["%s.%s" % (module_name, obj) for obj in allowed_list]

        # if 'methods' in self.options:
        #     _, methods = self.get_members(c, 'method', ['__init__'])

        #     self.content = ["~%s.%s" % (clazz, method) for method in methods if not method.startswith('_')]
        # if 'attributes' in self.options:
        #     _, attribs = self.get_members(c, 'attribute')
        #     self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]

        return Autosummary.run(self)


class AutoSummaryFromTag(Autosummary):
    """Usage:

    The following will print a summary of all object in module_name with 
    list of tags in there docstring (.. tags: list of tags) matching 
    tag_names: list of tags:

    .. autosumfromtag: module_name
        tag_names: list of tags
    
    """
    option_spec = {
        'tag_names': directives.unchanged,
    }

    required_arguments = 1

    def run(self):
        module_name = self.arguments[0]

        module = __import__(module_name, globals(), locals())
        tag_names = set(self.options['tag_names'].strip().split())

        self.content = []

        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if obj.__doc__ is None:
                continue
            match = re.search('\.\.\s*_tags\:([ \w]+)', obj.__doc__)
            if match is None:
                continue
            obj_tags = set(match.group(1).strip().split())
            if obj_tags == tag_names:
                self.content.append("%s.%s" % (module_name, obj_name))

        return Autosummary.run(self)


class AutoClassSummary(Autosummary):

    option_spec = {
        'methods': directives.unchanged,
        'attributes': directives.unchanged
    }

    required_arguments = 1

    @staticmethod
    def get_members(obj, typ, include_public=None):
        if not include_public:
            include_public = []
        items = []
        for name in dir(obj):
            try:
                documenter = get_documenter(safe_getattr(obj, name), obj)
            except AttributeError:
                continue
            if documenter.objtype == typ:
                items.append(name)
        public = [x for x in items if x in include_public or not x.startswith('_')]
        return public, items

    def run(self):
        clazz = self.arguments[0]
        try:
            (module_name, class_name) = clazz.rsplit('.', 1)
            m = __import__(module_name, globals(), locals(), [class_name])
            c = getattr(m, class_name)
            if 'methods' in self.options:
                _, methods = self.get_members(c, 'method', ['__init__'])

                self.content = ["~%s.%s" % (clazz, method) for method in methods if not method.startswith('_')]
            if 'attributes' in self.options:
                _, attribs = self.get_members(c, 'attribute')
                self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]
        finally:
            return super(AutoClassSummary, self).run()

def setup(app):
    app.add_directive('autosumfromlist', AutoSummaryFromList)
    app.add_directive('autosumfromtag', AutoSummaryFromTag)
    app.add_directive('autoclasssummary', AutoClassSummary)
