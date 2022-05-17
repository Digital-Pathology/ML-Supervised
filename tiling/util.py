
import os


def open_file(filepath, mode='r', binary_mode=False):
    if mode in ['w', 'x']:
        mode = 'w' if os.path.exists(filepath) else 'x'
    if binary_mode:
        mode += 'b'
    return open(filepath, mode)
