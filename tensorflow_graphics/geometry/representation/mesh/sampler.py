#Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Computes a weighted point sampling of a triangular mesh.

This op computes a uniform sampling of points on the surface of the mesh.
Points are sampled from the surface of each triangle using a uniform
distribution, proportional to a specified face density (e.g. face area).

Uses the approach mentioned in the TOG 2002 paper "Shape distributions"
(https://dl.acm.org/citation.cfm?id=571648)
to generate random barycentric coordinates.

For an example usage of this op to improve mesh reconstruction, see this ICML'19
paper: "GEOMetrics: Exploiting Geometric Structure for Graph-Encoded Objects"
(https://arxiv.org/abs/1901.11461)

Op is differentiable w.r.t mesh vertex positions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.representation import triangle as tfg_triangle
from tensorflow_graphics.geometry.representation.mesh import normals as tfg_normals
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import shape


def triangle_area(v0, v1, v2, name=None):
  """Computes triangle areas.

  Note:
    Computed triangle area = 0.5 * | e1 x e2 | where e1 and e2 are edges
      of triangle.

    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    v0: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the first vertex of a triangle.
    v1: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the second vertex of a triangle.
    v2: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the third vertex of a triangle.
    name: A name for this op. Defaults to "triangle_area".

  Returns:
    A tensor of shape `[A1, ..., An, 1]`, where the last dimension represents
      a normalized vector.
  """
  with tf.compat.v1.name_scope(name, "triangle_area"):
    v0 = tf.convert_to_tensor(value=v0)
    v1 = tf.convert_to_tensor(value=v1)
    v2 = tf.convert_to_tensor(value=v2)

    normals = tfg_triangle.normal(v0, v1, v2, normalize=False)
    areas = 0.5 * tf.linalg.norm(tensor=normals, axis=-1)
    return areas


def _random_categorical_sample(num_samples,
                               weights,
                               seed=None,
                               stateless=False,
                               name=None,
                               sample_dtype=tf.int32):
  """Sample from a random categorical distribution with general batch dims.

  Args:
    num_samples: A 0-D int32 denoting number of samples to generate per mesh.
    weights: A `float` tensor of shape `[A1, ..., An, F]` where F is number of
      faces. All weights should be > 0.
    seed: Optional random seed, value depends on stateless.
    stateless: Optional flag to use stateless random sampler. If stateless=True,
      then seed must be provided as shape `[2]` int tensor. Stateless random
      sampling is useful for testing to generate same sequence across calls.
    name: Name for op. Defaults to 'generate_random_face_ids'
    sample_dtype: Type of output samples.

  Returns:
    A 'sample_dtype' tensor of shape `[num_samples, A1, ..., An]`.
  """
  with tf.compat.v1.name_scope(name, "random_categorical_sample"):
    logits = tf.math.log(weights)
    num_faces = tf.shape(input=logits)[-1]
    batches = tf.shape(input=logits)[:-1]
    logits_2d = tf.reshape(logits, [-1, num_faces])
    if stateless:
      seed = tf.convert_to_tensor(value=seed)
      shape.check_static(
          tensor=seed, tensor_name="seed", has_dim_equals=(-1, 2))
      sample_fn = tf.random.stateless_categorical
    else:
      sample_fn = tf.random.categorical
    draws = sample_fn(
        logits=logits_2d,
        num_samples=num_samples,
        dtype=sample_dtype,
        seed=seed)
    samples = tf.reshape(
        tf.transpose(a=draws),
        shape=tf.concat([[num_samples], batches], axis=0))
    return samples


def generate_random_face_ids(num_samples,
                             face_weights,
                             seed=None,
                             stateless=False,
                             name=None):
  """Generate a sample of face ids given per face probability.

  In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    num_samples: A 0-D int32 denoting number of samples to generate per mesh.
    face_weights: A `float` tensor of shape `[A1, ..., An, F]` where F is
      number of faces. All weights should be > 0.
    seed: Optional random seed.
    stateless: Optional flag to use stateless random sampler. If
      stateless=True, then seed must be provided as shape `[2]` int tensor.
      Stateless random sampling is useful for testing to generate same sequence
      across calls.
    name: Name for op. Defaults to 'generate_random_face_ids'

  Returns:
    A `int32` tensor of shape `[A1, ..., An, num_samples]` denoting face ids
      sampled.
  """
  with tf.compat.v1.name_scope(name, "generate_random_face_ids"):
    num_samples = tf.convert_to_tensor(value=num_samples)
    face_weights = tf.convert_to_tensor(value=face_weights)
    shape.check_static(tensor=face_weights, has_rank_greater_than=0)
    shape.check_static(tensor=num_samples, has_rank=0)

    face_weights = asserts.assert_all_above(face_weights, minval=0.0)
    eps = asserts.select_eps_for_division(face_weights.dtype)
    face_weights = face_weights + eps
    sampled_face_ids = _random_categorical_sample(
        num_samples=num_samples,
        weights=face_weights,
        seed=seed,
        stateless=stateless)
    tgt_dims = tf.concat((tf.range(1, sampled_face_ids.shape.ndims), [0]),
                         axis=0)
    sampled_face_ids = tf.transpose(a=sampled_face_ids, perm=tgt_dims)
    return sampled_face_ids


def generate_random_barycentric(num_samples,
                                batch_shape=None,
                                dtype=tf.dtypes.float32,
                                seed=None,
                                stateless=False,
                                name=None):
  """Generate random barycentric coordinates.

  Args:
    num_samples: A 0-D `int` tensor denoting number of samples to generate
      per batch item.
    batch_shape: An optional `int` tuple denoting batch shape.
    dtype: Optional type of generated barycentric coordinates.
    seed: An optional random seed.
    stateless: Optional flag to use stateless random sampler. If
      stateless=True, then seed must be provided as shape `[2]` int tensor.
      Stateless random sampling is useful for testing to generate same sequence
      across calls.
    name: Name for op. Defaults to 'generate_random_barycentric'

  Returns:
    A `dtype` tensor of shape [A1, ..., An, num_samples, 3], where
      batch_shape = [A1, ..., An] are optional dimensions.

  """
  with tf.compat.v1.name_scope(name, "generate_random_barycentric"):
    num_samples = tf.convert_to_tensor(value=num_samples)
    shape.check_static(tensor=num_samples, has_rank=0)
    sample_shape = tf.concat(([num_samples], [2]), axis=0)

    if batch_shape is not None:
      batch_shape = tf.convert_to_tensor(value=batch_shape)
      shape.check_static(tensor=batch_shape, has_rank=1)
      sample_shape = tf.concat((batch_shape, sample_shape), axis=0)

    if stateless:
      seed = tf.convert_to_tensor(value=seed)
      shape.check_static(
          tensor=seed, tensor_name="seed", has_dim_equals=(-1, 2))
      sample_fn = tf.random.stateless_uniform
    else:
      sample_fn = tf.random.uniform
    r = sample_fn(
        shape=sample_shape, minval=0.0, maxval=1.0, dtype=dtype, seed=seed)
    r1 = tf.sqrt(r[..., 0])
    r2 = r[..., 1]
    barycentric = tf.stack((1-r1, r1 * (1-r2), r1 * r2), axis=-1)
    return barycentric


def weighted_random_sample_tri_mesh(vertices,
                                    faces,
                                    num_samples,
                                    face_weights,
                                    seed=None,
                                    stateless=False,
                                    name=None):
  """Performs a face probability weighted random sampling of a tri mesh.

  In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    vertices: A `float` tensor of shape `[A1, ..., An, V, D]`, where V is the
       number of vertices, and D is dimensionality of each vertex.
    faces: A `int` tensor of shape `[A1, ..., An, F, 3]`, where F is the
      number of faces.
    num_samples: A `int` 0-D tensor denoting number of samples to be drawn from
      each mesh.
    face_weights: A `float` tensor of shape ``[A1, ..., An, F]`, denoting
      sampling probability of each face, where F is the number of faces.
    seed: Optional random seed.
    stateless: Optional flag to use stateless random sampler. If
      stateless=True, then seed must be provided as shape `[2]` int tensor.
      Stateless random sampling is useful for testing to generate same sequence
      across calls.
    name: Name for op, defaults to 'weighted_random_sample_tri_mesh'

  Returns:
    sample_pts: A `float` tensor of shape `[A1, ..., An, num_samples, D]`,
      where D is dimensionality of each sampled point.
    sample_faceids: A `int` tensor of shape `[A1, ..., An, num_samples]`.
  """
  with tf.compat.v1.name_scope(name, "weighted_random_sample_tri_mesh"):
    faces = tf.convert_to_tensor(value=faces)
    vertices = tf.convert_to_tensor(value=vertices)
    face_weights = tf.convert_to_tensor(value=face_weights)
    num_samples = tf.convert_to_tensor(value=num_samples)

    shape.check_static(
        tensor=vertices, tensor_name="vertices", has_rank_greater_than=1)
    shape.check_static(
        tensor=faces, tensor_name="faces", has_rank_greater_than=1)
    shape.check_static(
        tensor=face_weights,
        tensor_name="face_weights",
        has_rank_greater_than=0)
    shape.compare_batch_dimensions(
        tensors=(vertices, faces, face_weights),
        last_axes=(-3, -3, -2),
        broadcast_compatible=False)

    asserts.assert_all_above(face_weights, 0)

    batch_shape = None
    batch_dims = faces.shape.ndims - 2
    if batch_dims > 0:
      batch_shape = faces.shape.as_list()[:-2]

    sample_faceids = generate_random_face_ids(
        num_samples, face_weights, seed=seed, stateless=stateless)
    sample_vertids = tf.gather(faces, sample_faceids, batch_dims=batch_dims)
    sample_vertices = tf.gather(vertices, sample_vertids, batch_dims=batch_dims)
    barycentric = generate_random_barycentric(
        num_samples,
        batch_shape=batch_shape,
        dtype=vertices.dtype,
        seed=seed,
        stateless=stateless)
    barycentric = tf.expand_dims(barycentric, axis=-1)
    sample_pts = tf.math.multiply(sample_vertices, barycentric)
    sample_pts = tf.reduce_sum(input_tensor=sample_pts, axis=-2)
    return sample_pts, sample_faceids


def area_weighted_random_sample_tri_mesh(vertices,
                                         faces,
                                         num_samples,
                                         vertex_positions=None,
                                         seed=None,
                                         stateless=False,
                                         name=None):
  """Performs a face area weighted random sampling of a tri mesh.

  Args:
    vertices: A `float` tensor of shape `[A1, ..., An, V, D]`, where V is the
       number of vertices, and D is dimensionality of each vertex.
    faces: A `int` tensor of shape `[A1, ..., An, F, 3]`, where F is the
      number of faces.
    num_samples: A `int` 0-D tensor denoting number of samples to be drawn from
      each mesh.
    vertex_positions: An optional `float` tensor of shape `[A1, ..., An, V, 3]`,
      where V is the number of vertices. If None, then vertices[..., :3] is
      used as vertex positions.
    seed: Optional random seed.
    stateless: Optional flag to use stateless random sampler. If
      stateless=True, then seed must be provided as shape `[2]` int tensor.
      Stateless random sampling is useful for testing to generate same sequence
      across calls.
    name: Name for op.
  Returns:
    sample_pts: A `float` tensor of shape `[A1, ..., An, num_samples, D]`,
      where D is dimensionality of each sampled point.
    sample_faceids: A `int` tensor of shape `[A1, ..., An, num_samples]`.
  """
  with tf.compat.v1.name_scope(name, "weighted_random_sample_tri_mesh"):
    faces = tf.convert_to_tensor(value=faces)
    vertices = tf.convert_to_tensor(value=vertices)
    num_samples = tf.convert_to_tensor(value=num_samples)

    shape.check_static(
        tensor=vertices, tensor_name="vertices", has_rank_greater_than=1)
    shape.check_static(
        tensor=vertices, tensor_name="vertices", has_dim_greater_than=(-1, 2))

    if vertex_positions is not None:
      vertex_positions = tf.convert_to_tensor(value=vertex_positions)
    else:
      vertex_positions = vertices[..., :3]

    shape.check_static(
        tensor=vertex_positions,
        tensor_name="vertex_positions",
        has_rank_greater_than=1)
    shape.check_static(
        tensor=vertex_positions,
        tensor_name="vertex_positions",
        has_dim_equals=(-1, 3))

    tri_vertex_positions = tfg_normals.gather_faces(vertex_positions, faces)
    tri_areas = triangle_area(tri_vertex_positions[..., 0, :],
                              tri_vertex_positions[..., 1, :],
                              tri_vertex_positions[..., 2, :])
    return weighted_random_sample_tri_mesh(
        vertices,
        faces,
        num_samples,
        face_weights=tri_areas,
        seed=seed,
        stateless=stateless)
