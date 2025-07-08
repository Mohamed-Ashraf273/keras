import numpy as np

from keras import ops
from keras.src import testing


class GetItemOpsTest(testing.TestCase):
    def test_getitem(self):
        self.np_tensor = np.arange(24).reshape(2, 3, 4)
        self.tensor = ops.convert_to_tensor(self.np_tensor)

        t = self.tensor[1]
        n = self.np_tensor[1]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1, 2, 3]
        n = self.np_tensor[1, 2, 3]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1:2]
        n = self.np_tensor[1:2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1:2, 2:3, 3:4]
        n = self.np_tensor[1:2, 2:3, 3:4]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1:2, None]
        n = self.np_tensor[1:2, None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1:2, 2:3, ...]
        n = self.np_tensor[1:2, 2:3, ...]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1:2, ..., 3:4]
        n = self.np_tensor[1:2, ..., 3:4]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[:, 2, ...]
        n = self.np_tensor[:, 2, ...]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[None, ..., 3:4, None]
        n = self.np_tensor[None, ..., 3:4, None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1:2:None]
        n = self.np_tensor[1:2:None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[:, 2]
        n = self.np_tensor[:, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[None]
        n = self.np_tensor[None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[None, None]
        n = self.np_tensor[None, None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[...]
        n = self.np_tensor[...]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[..., 1]
        n = self.np_tensor[..., 1]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[..., 1, 2]
        n = self.np_tensor[..., 1, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[..., -1, 2]
        n = self.np_tensor[..., -1, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[..., -1:-2, 2]
        n = self.np_tensor[..., -1:-2, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[..., None, None]
        n = self.np_tensor[..., None, None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[None, ..., None]
        n = self.np_tensor[None, ..., None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1, 2, None, ..., None]
        n = self.np_tensor[1, 2, None, ..., None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[None, ..., 1, 2]
        n = self.np_tensor[None, ..., 1, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1, None, 2]
        n = self.np_tensor[1, None, 2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        index_tensor = ops.convert_to_tensor(np.array(1, dtype=np.int32))
        t = self.tensor[index_tensor]
        n = self.np_tensor[ops.convert_to_numpy(index_tensor)]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        index_tensor = ops.convert_to_tensor(np.array(1, dtype=np.int32))
        t = self.tensor[index_tensor, 2, None]
        n = self.np_tensor[ops.convert_to_numpy(index_tensor), 2, None]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        index_tensor = ops.convert_to_tensor(np.array(-2, dtype=np.int32))
        t = self.tensor[index_tensor, 1]
        n = self.np_tensor[ops.convert_to_numpy(index_tensor), 1]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        index_tensor = ops.convert_to_tensor(np.array(-1, dtype=np.int32))
        t = self.tensor[-2, index_tensor]
        n = self.np_tensor[-2, ops.convert_to_numpy(index_tensor)]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        # Negative indexing
        t = self.tensor[-1]
        n = self.np_tensor[-1]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[1, -1, -2]
        n = self.np_tensor[1, -1, -2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        # Slicing with step
        t = self.tensor[::2]
        n = self.np_tensor[::2]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        # Mixed slices and integers
        t = self.tensor[1, :, 1:4]
        n = self.np_tensor[1, :, 1:4]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))

        t = self.tensor[:, 1:2, 3]
        n = self.np_tensor[:, 1:2, 3]
        self.assertEqual(t.shape, n.shape)
        self.assertTrue(np.array_equal(ops.convert_to_numpy(t), n))
