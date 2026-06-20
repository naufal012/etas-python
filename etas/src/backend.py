"""
Computational backend dispatcher for ETAS model.
Allows dynamic switching between CPU (NumPy) and GPU (CuPy).
"""

import numpy as np

_ENGINE = 'cpu'
_XP = np
_SPECIAL = None

try:
    import scipy.special
    _SPECIAL = scipy.special
except ImportError:
    pass

def set_engine(engine='cpu'):
    """
    Set the computational engine to use for mathematical operations.
    
    Parameters
    ----------
    engine : str
        'cpu' to use NumPy, 'gpu' to use CuPy.
    """
    global _ENGINE, _XP, _SPECIAL
    
    engine = engine.lower()
    if engine == 'gpu':
        try:
            import cupy as cp
            _ENGINE = 'gpu'
            _XP = cp
            
            try:
                import cupyx.scipy.special
                _SPECIAL = cupyx.scipy.special
            except ImportError:
                pass
                
            print("ETAS backend set to GPU (CuPy).")
        except ImportError:
            import warnings
            warnings.warn("CuPy is not installed or CUDA is unavailable. "
                          "Falling back to CPU backend (NumPy).")
            _ENGINE = 'cpu'
            _XP = np
            try:
                import scipy.special
                _SPECIAL = scipy.special
            except ImportError:
                pass
    elif engine == 'cpu':
        _ENGINE = 'cpu'
        _XP = np
        try:
            import scipy.special
            _SPECIAL = scipy.special
        except ImportError:
            pass
    else:
        raise ValueError("Engine must be 'cpu' or 'gpu'")

def get_xp():
    """Get the active array module (numpy or cupy)."""
    return _XP

def get_engine():
    """Get the name of the active engine ('cpu' or 'gpu')."""
    return _ENGINE

def get_special():
    """Get the active scipy special module (scipy.special or cupyx.scipy.special)."""
    return _SPECIAL
