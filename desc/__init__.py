import colorama
from termcolor import colored
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

colorama.init()

_BANNER = """
 ____  ____  _____   ___ 
|  _ \| ___|/  ___|/ ___|
| | \ | |_  | (__ | |
| | | |  _| \___ \| |
| |_/ | |__  ___) | |___ 
|____/|____||____/ \____|
                         
"""

BANNER = colored(_BANNER, 'magenta')
