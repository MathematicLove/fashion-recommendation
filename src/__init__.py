import os
# Allow duplicate OpenMP runtimes so torch does not segfault on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")