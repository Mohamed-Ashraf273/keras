import numpy as np
import openvino.runtime.opset14 as ov_opset
from openvino import Type

from keras.src.backend.config import floatx
from keras.src.backend.openvino.core import OPENVINO_DTYPES
from keras.src.backend.openvino.core import OpenVINOKerasTensor
from keras.src.backend.openvino.core import convert_to_numpy
from keras.src.backend.openvino.core import get_ov_output
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed.data)
    normal_const = rng.normal(size=shape, loc=mean, scale=stddev).astype(dtype)
    return OpenVINOKerasTensor(ov_opset.constant(normal_const).output(0))


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    ov_type = OPENVINO_DTYPES[dtype]
    seed = draw_seed(seed)
    if isinstance(seed, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed)
    else:
        seed1, seed2 = draw_seed(seed).data
    minval_const = ov_opset.constant(minval, dtype=dtype)
    maxval_const = ov_opset.constant(maxval, dtype=dtype)
    if isinstance(shape, tuple):
        shape = list(shape)
    output_shape_const = ov_opset.constant(shape, dtype=Type.i32)
    random_uniform = ov_opset.random_uniform(
        output_shape_const, minval_const, maxval_const, ov_type, seed1, seed2
    )
    return OpenVINOKerasTensor(random_uniform.output(0))


def categorical(logits, num_samples, dtype="int64", seed=None):
    def get_shape_dims(x):
        shape = ov_opset.shape_of(x, Type.i32)
        rank_tensor = ov_opset.shape_of(shape, Type.i32)
        rank_scalar = ov_opset.squeeze(
            rank_tensor, ov_opset.constant([0], Type.i32)
        )
        indices = ov_opset.range(
            ov_opset.constant(0, Type.i32),
            rank_scalar,
            ov_opset.constant(1, Type.i32),
            output_type=Type.i32,
        )
        return ov_opset.gather(shape, indices, axis=0)

    dtype = dtype or "int64"
    ov_dtype = OPENVINO_DTYPES[dtype]
    logits = get_ov_output(logits)
    probs = ov_opset.softmax(logits, axis=-1)
    cumsum_probs = ov_opset.cumsum(probs, ov_opset.constant(-1, dtype="int32"))
    shape = get_shape_dims(logits)
    rank_tensor = ov_opset.shape_of(shape, Type.i32)
    rank = ov_opset.squeeze(rank_tensor, ov_opset.constant([0], dtype=Type.i32))
    rank_minus_1 = ov_opset.subtract(rank, ov_opset.constant(1, dtype=Type.i32))
    indices = ov_opset.range(
        ov_opset.constant(0, dtype=Type.i32),
        rank_minus_1,
        ov_opset.constant(1, dtype=Type.i32),
        output_type=Type.i32,
    )
    batch_shape = ov_opset.gather(shape, indices, axis=0)
    final_shape = ov_opset.concat(
        [batch_shape, ov_opset.constant([num_samples], dtype=Type.i32)], axis=0
    )
    seed_tensor = draw_seed(seed)
    if isinstance(seed_tensor, OpenVINOKerasTensor):
        seed1, seed2 = convert_to_numpy(seed_tensor)
    else:
        seed1, seed2 = seed_tensor.data
    rand = ov_opset.random_uniform(
        final_shape,
        ov_opset.constant(0.0, dtype=probs.get_element_type()),
        ov_opset.constant(1.0, dtype=probs.get_element_type()),
        probs.get_element_type(),
        seed1,
        seed2,
    )
    rand = ov_opset.unsqueeze(rand, [-1])
    cumsum_probs = ov_opset.unsqueeze(
        cumsum_probs, ov_opset.constant([1], dtype=Type.i32)
    )
    greater = ov_opset.greater(rand, cumsum_probs)
    samples = ov_opset.reduce_sum(
        ov_opset.convert(greater, Type.i32),
        ov_opset.constant([-1], dtype=Type.i32),
    )
    samples = ov_opset.convert(samples, ov_dtype)
    return OpenVINOKerasTensor(samples.output(0))


def randint(shape, minval, maxval, dtype="int32", seed=None):
    raise NotImplementedError(
        "`randint` is not supported with openvino backend"
    )


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    seed = draw_seed(seed)
    rng = np.random.default_rng(seed.data)

    lower_bound = mean - 2 * stddev
    upper_bound = mean + 2 * stddev

    flat_shape = np.prod(shape)
    random_numbers = np.empty(0)

    # loop until we have enough valid numbers to fill our desired shape
    while random_numbers.shape[0] < flat_shape:
        # Generate a batch of random numbers from a normal distribution
        batch = rng.normal(loc=mean, scale=stddev, size=flat_shape)

        # Filter the numbers to keep only those within the specified bounds
        valid = batch[(batch >= lower_bound) & (batch <= upper_bound)]

        # Append the valid numbers to the result array
        random_numbers = np.append(random_numbers, valid)

    # Truncate the result array to the desired size and reshape it
    np_array_res = random_numbers[:flat_shape].astype(dtype).reshape(shape)
    return OpenVINOKerasTensor(ov_opset.constant(np_array_res).output(0))


def dropout(inputs, rate, noise_shape=None, seed=None):
    raise NotImplementedError(
        "`dropout` is not supported with openvino backend"
    )


def shuffle(x, axis=0, seed=None):
    raise NotImplementedError(
        "`shuffle` is not supported with openvino backend"
    )


def gamma(shape, alpha, dtype=None, seed=None):
    raise NotImplementedError("`gamma` is not supported with openvino backend")


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    raise NotImplementedError(
        "`binomial` is not supported with openvino backend"
    )


def beta(shape, alpha, beta, dtype=None, seed=None):
    raise NotImplementedError("`beta` is not supported with openvino backend")
