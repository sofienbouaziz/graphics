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
"""Tests for uniform mesh sampler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.representation.mesh import sampler
from tensorflow_graphics.geometry.representation.mesh.tests import mesh_test_utils
from tensorflow_graphics.util import test_case


class MeshSamplerTest(test_case.TestCase):

  def setUp(self):
    """Setup default parameters."""
    super(MeshSamplerTest, self).setUp()
    self._test_sigma_compare_tolerance = 4.0

  def compare_poisson_equiv(self, expected, actual):
    """Poisson-distributed random check."""
    delta = np.sqrt(expected) * self._test_sigma_compare_tolerance
    self.assertAllClose(expected, actual, atol=delta)

  @parameterized.parameters(
      (((4, 3), (5, 3), ()), (tf.float32, tf.int32, tf.int32)),
      (((None, 3), (None, 3), ()), (tf.float32, tf.int32, tf.int32)),
      (((3, None, 3), (3, None, 3), ()), (tf.float32, tf.int32, tf.int32)),
      # Test for vertex attributes + positions
      (((3, 6, 5), (3, 5, 3), (), (3, 6, 3)),
       (tf.float32, tf.int32, tf.int32, tf.float32)),
  )
  def test_sampler_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        sampler.area_weighted_random_sample_tri_mesh, shapes, dtypes)

  @parameterized.parameters(
      ("vertices must have a rank greater than 1.", (3,), (None, 3), ()),
      ("vertices must have greater than 2 dimensions in axis -1.", (5, 2),
       (None, 3), ()),
      ("vertex_positions must have exactly 3 dimensions in axis -1.", (5, 3),
       (None, 3), (), (3, 2)),
  )
  def test_sampler_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        sampler.area_weighted_random_sample_tri_mesh, error_msg, shapes)

  def test_sampler_negative_weights(self):
    """Test for exception with negative weights."""
    vertices, faces = mesh_test_utils.create_single_tri_mesh()
    face_wts = np.array([-0.3, 0.1, 0.5, 0.6], dtype=np.float32)
    num_samples = 10
    error_msg = "Condition x >= y did not hold."
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, error_msg):
      sampler.weighted_random_sample_tri_mesh(
          vertices, faces, num_samples, face_weights=face_wts)

  def test_sample_distribution(self):
    """Rough test for uniform sample distribution."""
    vertices, faces = mesh_test_utils.create_single_tri_mesh()
    vertices = np.repeat(np.expand_dims(vertices, axis=0), 3, axis=0)
    faces = np.repeat(np.expand_dims(faces, axis=0), 3, axis=0)
    num_samples = 5000
    sample_pts, _ = sampler.area_weighted_random_sample_tri_mesh(
        vertices, faces, num_samples)

    for i in range(3):
      samples = sample_pts[i, ...]
      self.assertEqual(samples.shape[-2], num_samples)
      # Test distribution in 4 quadrants of [0,1]x[0,1]
      v = samples[:, :2] < [0.5, 0.5]
      not_v = tf.logical_not(v)
      quad00 = tf.math.count_nonzero(tf.reduce_all(input_tensor=v, axis=-1))
      quad11 = tf.math.count_nonzero(tf.reduce_all(input_tensor=not_v, axis=-1))
      quad01 = tf.math.count_nonzero(tf.reduce_all(
          input_tensor=tf.stack((v[:, 0], not_v[:, 1]), axis=1), axis=-1))
      quad10 = tf.math.count_nonzero(tf.reduce_all(
          input_tensor=tf.stack((not_v[:, 0], v[:, 1]), axis=1), axis=-1))
      counts = tf.stack((quad00, quad01, quad10, quad11), axis=0)
      expected = np.array(
          [num_samples / 2, num_samples / 4, num_samples / 4, 0],
          dtype=np.float32)
      self.compare_poisson_equiv(expected, counts)

  def test_face_distribution(self):
    """Rough test for distribution of point face indices."""
    vertices, faces = mesh_test_utils.create_square_tri_mesh()
    num_samples = 1000
    _, sample_faces = sampler.area_weighted_random_sample_tri_mesh(
        vertices, faces, num_samples)

    # All points should be approx poisson distributed among the 4 faces.
    self.assertEqual(sample_faces.shape[0], num_samples)
    num_faces = faces.shape[0]
    expected = np.array([num_samples / num_faces] * num_faces, dtype=np.intp)
    self.compare_poisson_equiv(expected, tf.math.bincount(sample_faces))

  def test_sampler_jacobian_random(self):
    """Test the Jacobian of the vertex normals function."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    vertex_axis = np.array(((0., 0., 1), (1., 0., 0.), (0., 1., 0.),
                            (0., 0., -1.), (-1., 0., 0.), (0., -1., 0.)),
                           dtype=np.float32)
    vertex_axis = vertex_axis.reshape([1] * tensor_vertex_size + [6, 3])
    faces = np.array(((0, 1, 2), (0, 2, 4), (0, 4, 5), (0, 5, 1), (3, 2, 1),
                      (3, 4, 2), (3, 5, 4), (3, 1, 5)),
                     dtype=np.int32)
    faces = faces.reshape([1] * tensor_vertex_size + [8, 3])
    index_init = np.tile(faces, tensor_out_shape + [1, 1])
    vertex_scale = np.random.uniform(0.5, 5., tensor_out_shape + [1] * 2)
    vertex_init = vertex_axis * vertex_scale
    vertex_tensor = tf.convert_to_tensor(value=vertex_init)
    index_tensor = tf.convert_to_tensor(value=index_init)

    num_samples = np.random.randint(10, 100)
    sample_pts, _ = sampler.area_weighted_random_sample_tri_mesh(
        vertex_tensor, index_tensor, num_samples, seed=[0, 1], stateless=True)

    self.assert_jacobian_is_correct(
        vertex_tensor, vertex_init, sample_pts, atol=1e-4, delta=1e-4)


if __name__ == "__main__":
  test_case.main()
