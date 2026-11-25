"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

"""
Configure Python paths for the GeoMapNet repository.

This file ensures that the original imports work independently
of the current working directory.
"""

import sys
from pathlib import Path

# Directory containing this file:
# geomapnet/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent

# GeoMapNet root:
# geomapnet/
GEOMAPNET_DIR = SCRIPT_DIR.parent

# Add GeoMapNet root to Python module search path.
GEOMAPNET_PATH = str(GEOMAPNET_DIR)

if GEOMAPNET_PATH not in sys.path:
    sys.path.insert(0, GEOMAPNET_PATH)
