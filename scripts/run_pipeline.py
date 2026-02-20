# Master script: calls everything in sequence

"""This package contains your two data modules. A good use of this init is to surface the two functions you will call most often from outside, so your pipeline script has clean imports.
# data/__init__.py
from .label_parser import build_class_dictionary, parse_pseudo_yolo_labels
from .dataset_splitter import make_splits
Now in your pipeline script you can write:
from scribal_spotting.data import build_class_dictionary, make_splits
instead of specifying which file each lives in.
"""
