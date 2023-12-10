#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:53:55 2023

@author: renatotronofigueras
"""

from os.path import dirname, basename, isfile, join
import glob

__all__ = []
modules = glob.glob(join(dirname(__file__), "*.py"))

for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        __all__.append(basename(f)[:-3])

from . import *