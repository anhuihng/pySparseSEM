"""
def loadSEMlib():
=======================
INPUT ARGUMENTS:
                NONE
=======================
OUTPUT ARGUMENTS:
semlib          Returns a glmlib object with methods that are equivalent
                to the C functions in /src
=======================
"""
import ctypes
import os
sparseSEM_so = 'sparseSEM.cpython-310-darwin.so'

sparseSEM_so = os.path.dirname(__file__) + '/sparseSEM.cpython-310-darwin.so'
sparseSEM_dll = os.path.dirname(__file__) + '/sparseSEM.cpython-310-darwin.dll'

def loadSEMlib():
    if os.name == 'posix':
        semlib = ctypes.cdll.LoadLibrary(sparseSEM_so)
        return(semlib)
    elif os.name == 'nt':
        # this does not currently work
        raise ValueError('loadSEMlib does not currently work for windows')
        # semlib = ctypes.windll.LoadLibrary(sparseSEM_dll)
    else:
        raise ValueError('loadSEMlib not yet implemented for non-posix OS')