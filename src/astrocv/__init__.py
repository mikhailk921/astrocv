"""
Initialization of package: just export most useful things
"""

## @package astrocv
#  @brief Python implementation image processing
#
# Some description of architecture


from .tracker.Tracker import *
from .processing import *
from .ganerate import *
from .search import *

set_threads_count(1)
