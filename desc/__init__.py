from termcolor import colored
import colorama
colorama.init()

__version__ = '0.2.0'

_BANNER = """
 ____  ____  _____   ___ 
|  _ \| ___|/  ___|/ ___|
| | \ | |_  | (__ | |
| | | |  _| \___ \| |
| |_/ | |__  ___) | |___ 
|____/|____||____/ \____|
                         
"""

BANNER = colored(_BANNER, 'magenta')
