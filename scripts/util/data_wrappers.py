"""
Wrappers to manage datasets in a consistent manner.
We want to be able to get them all as tensors with epoch limiters so that
we don't have to worry about feeds etc.
"""
import rnndatasets
