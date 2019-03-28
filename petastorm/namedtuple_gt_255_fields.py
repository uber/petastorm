#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys as _sys
from keyword import iskeyword as _iskeyword

_class_template = """\
from builtins import property as _property, tuple as _tuple
from operator import itemgetter as _itemgetter
from collections import OrderedDict
from petastorm.namedtuple_gt_255_fields import _restore_namedtuple_gt_255_fields
class {typename}(tuple):
    '{typename}({arg_list})'
    __slots__ = ()
    _fields = {field_names!r}
    def __new__(_cls, *args, **kwargs):
        'Create new instance of {typename}({arg_list})'
        values = []
        missings = []
        for index, field in enumerate(_cls._fields):
            if index < len(args):
                values.append(args[index])
                if field in kwargs:
                    raise TypeError('__new__() got multiple values for argument %r' % field)
            elif field in kwargs:
                values.append(kwargs[field])
            else:
                values.append(None)
                missings.append(field)
        count_missings = len(missings)
        if count_missings > 0:
            last_missing = ' and %s' % missings[-1] if count_missings > 1 else ''
            missings = missings[:-1] if count_missings > 1 else missings
            raise TypeError('__new__() missing %d required positional argument%s: %s%s' %
                            (count_missings, 's' if count_missings > 1 else '',
                            ', '.join(missings), last_missing))
        return _tuple.__new__(_cls, values)
    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        'Make a new {typename} object from a sequence or iterable'
        result = new(cls, iterable)
        if len(result) != {num_fields:d}:
            raise TypeError('Expected {num_fields:d} arguments, got %d' % len(result))
        return result
    def _replace(_self, **kwds):
        'Return a new {typename} object replacing specified fields with new values'
        result = _self._make(map(kwds.pop, {field_names!r}, _self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % list(kwds))
        return result
    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + '({repr_fmt})' % self
    def _asdict(self):
        'Return a new OrderedDict which maps field names to their values.'
        return OrderedDict(zip(self._fields, self))
    def __getnewargs__(self):
        'Return self as a plain tuple. Used by copy and pickle.'
        return tuple(self)
    def __reduce__(self):
        'Makes the namedtuple pickable.'
        cls = self.__class__
        return (_restore_namedtuple_gt_255_fields, (cls.__name__, cls._fields, tuple(self)))
{field_defs}
"""

_repr_template = '{name}=%r'

_field_template = '''\
    {name} = _property(_itemgetter({index:d}), doc='Alias for field number {index:d}')
'''


def _restore_namedtuple_gt_255_fields(typename, fields, values):
    """Creates an namedtuple_gt_255_fields objects along based its __reduce__ description.

    The __reduce__ protocol to support pickling requires:
    - a callable method that returns the pickled object on restore
    - a tuple of its arguments.

    This function is called by the __reduce__ method of a generated namedtuple_gt_255_fields
    to recreate the inital object.
    """
    return namedtuple_gt_255_fields(typename, fields)(*values)


def namedtuple_gt_255_fields(typename, field_names, verbose=False, rename=False, module=None):
    """Returns a new subclass of tuple with named fields.
    >>> Point = namedtuple2('Point', ['x', 'y'])
    >>> Point.__doc__                   # docstring for the new class
    'Point(x, y)'
    >>> p = Point(11, y=22)             # instantiate with positional args or keywords
    >>> p[0] + p[1]                     # indexable like a plain tuple
    33
    >>> x, y = p                        # unpack like a regular tuple
    >>> x, y
    (11, 22)
    >>> p.x + p.y                       # fields also accessible by name
    33
    >>> d = p._asdict()                 # convert to a dictionary
    >>> d['x']
    11
    >>> Point(**d)                      # convert from a dictionary
    Point(x=11, y=22)
    >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    Point(x=100, y=22)
    """

    # Validate the field names.  At the user's option, either generate an error
    # message or automatically replace the field name with a valid name.
    if isinstance(field_names, str):
        field_names = field_names.replace(',', ' ').split()
    field_names = list(map(str, field_names))
    typename = str(typename)
    if rename:
        seen = set()
        for index, name in enumerate(field_names):
            if (not name.isidentifier()
                    or _iskeyword(name)
                    or name.startswith('_')
                    or name in seen):
                field_names[index] = '_%d' % index
            seen.add(name)
    for name in [typename] + field_names:
        if type(name) is not str:
            raise TypeError('Type names and field names must be strings')
        if not name.isidentifier():
            raise ValueError('Type names and field names must be valid '
                             'identifiers: %r' % name)
        if _iskeyword(name):
            raise ValueError('Type names and field names cannot be a '
                             'keyword: %r' % name)
    seen = set()
    for name in field_names:
        if name.startswith('_') and not rename:
            raise ValueError('Field names cannot start with an underscore: '
                             '%r' % name)
        if name in seen:
            raise ValueError('Encountered duplicate field name: %r' % name)
        seen.add(name)

    # Fill-in the class template
    class_definition = _class_template.format(
        typename=typename,
        field_names=tuple(list(field_names)),
        num_fields=len(field_names),
        arg_list=repr(list(field_names)).replace("'", "")[1:-1],
        repr_fmt=', '.join(_repr_template.format(name=name)
                           for name in field_names),
        field_defs='\n'.join(_field_template.format(index=index, name=name)
                             for index, name in enumerate(field_names))
    )

    # Execute the template string in a temporary namespace and support
    # tracing utilities by setting a value for frame.f_globals['__name__']
    namespace = dict(__name__='namedtuple_gt_255_fields_%s' % typename)
    # pylint: disable=W0122
    exec(class_definition, namespace)
    result = namespace[typename]
    result._source = class_definition
    if verbose:
        print(result._source)

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in environments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython), or where the user has
    # specified a particular module.
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module

    return result
