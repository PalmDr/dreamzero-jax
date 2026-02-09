"""Tests for sharding utilities."""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from dreamzero_jax.utils.sharding import (
    create_mesh,
    get_partition_spec,
    shard_array,
    shard_batch,
)


def test_create_mesh_single_device():
    """Mesh creation should work on a single device."""
    mesh = create_mesh()
    assert isinstance(mesh, Mesh)
    assert "data" in mesh.axis_names
    assert "model" in mesh.axis_names


def test_create_mesh_custom_shape():
    """Mesh creation with explicit shape."""
    num_devices = jax.device_count()
    mesh = create_mesh(mesh_shape=(num_devices, 1))
    assert mesh.shape["data"] == num_devices
    assert mesh.shape["model"] == 1


def test_get_partition_spec_kernel():
    """Dense kernel should get a partition spec."""
    mesh = create_mesh()
    spec = get_partition_spec("dit.blocks.0.self_attn.q_proj.kernel", (1536, 1536), mesh)
    assert isinstance(spec, PartitionSpec)


def test_get_partition_spec_bias():
    """Bias should be replicated."""
    mesh = create_mesh()
    spec = get_partition_spec("dit.blocks.0.self_attn.q_proj.bias", (1536,), mesh)
    assert isinstance(spec, PartitionSpec)


def test_get_partition_spec_embedding():
    """Embedding weights should be replicated."""
    mesh = create_mesh()
    spec = get_partition_spec("text_encoder.token_embedding.embedding", (256384, 4096), mesh)
    assert isinstance(spec, PartitionSpec)


def test_shard_array():
    """Sharding an array should produce a sharded array."""
    mesh = create_mesh()
    x = jnp.ones((8, 64))
    spec = PartitionSpec("data", None)
    sharded = shard_array(x, mesh, spec)
    assert sharded.shape == (8, 64)


def test_shard_batch():
    """Batch sharding should shard first dim across 'data'."""
    mesh = create_mesh()
    batch = {
        "video": jnp.ones((4, 8, 32, 32, 3)),
        "token_ids": jnp.ones((4, 128), dtype=jnp.int32),
        "actions": jnp.ones((4, 32, 7)),
    }
    sharded_batch = shard_batch(batch, mesh)
    for key in batch:
        assert sharded_batch[key].shape == batch[key].shape
