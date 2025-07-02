import numpy as np

from keras import ops


def test_triu_basic():
    x = np.arange(12).reshape(3, 4)
    expected = np.triu(x)
    result = ops.triu(x)
    print(ops.convert_to_numpy(result) == expected)


test_triu_basic()
