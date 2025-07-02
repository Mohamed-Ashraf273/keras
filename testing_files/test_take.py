import numpy as np

x = np.arange(24).reshape([1, 2, 3, 4])
indices = np.ones([1, 4, 1, 1], dtype=np.int32)
print(np.take_along_axis(x, indices, axis=1))
