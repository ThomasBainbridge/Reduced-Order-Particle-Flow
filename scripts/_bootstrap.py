"""Make the ``ropf`` package importable when running scripts directly.

Allows ``python scripts/generate_dataset.py`` to work straight from a clone
without first running ``pip install -e .`` (though that also works).
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
