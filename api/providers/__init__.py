"""
api/providers package init — re-exports everything from the original api/providers.py
module (which is shadowed on disk by this package directory).

Provider adapters live in api/providers/anthropic_adapter.py etc.
"""
import importlib.util as _ilu
import os as _os

_fp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "providers.py")
_spec = _ilu.spec_from_file_location("api._providers_impl", _fp)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_mod, _name)

del _ilu, _os, _fp, _spec, _mod, _name
