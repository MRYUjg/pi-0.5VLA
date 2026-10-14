#!/usr/bin/env python3
"""N0 compatibility entry: verify by default; create requires a new version.
Historical implementations retired; see maintenance/cleanup_20260929/.
"""
from n0_freeze import main

if __name__ == '__main__':
    raise SystemExit(main())
