"""
Clean working directory from artifacts of predictions
"""

import os


def cleanup() -> None:
    directory = 'tmp/'
    os.remove(directory + 'coords.csv')
    for file in [os.path.join(directory + 'faces/', f) for f in os.listdir(directory + 'faces/')]:
        os.remove(file)
    return None


if __name__ == "__main__":
    cleanup()
