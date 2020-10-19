"""The "single-point-of-truth" for package version and name

Contents of this file are assumed to be parsed by various programs to access
basic package info like version and name.
"""

## @file info.py
#  @brief Single point-of-truth for package metainfo
#
# The most simple usage of this file is shown below
# ~~~~~~~~~~~{.py}
# INFO_DIR = "astrocv"  # absolute or relative path to directory with this file
# info = {}  # placeholder for package information
# with open(INFO_DIR + "/info.py", "r") as finfo:
#     exec(finfo.read(), info)
# print("Package name: " + info["__package_name__"])
# print("Package version:" + info["__version__"])
# ~~~~~~~~~~~

__package_name__ = "astrocv"

__version_info__ = (0, 1, 0)

__version__ = ".".join(map(str, __version_info__))
__version_major__ = __version_info__[0]
__version_minor__ = __version_info__[1]
__version_patch__ = __version_info__[2]
__version_id__ = ((__version_major__ << 24) |
                  (__version_minor__ << 16) |
                  (__version_patch__ << 0))
