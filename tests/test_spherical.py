import numpy as np
import pytest

from llnltofi._spherical import cart2sph, sph2cart, geo2sph, sph2geo, cart2polar


def test_cart2sph_identity():
    cart = np.array([1.0, 0.0, 0.0])
    sph = cart2sph(cart)
    assert abs(sph[0] - 1.0) < 1e-12  # r
    assert abs(sph[1] - 0.0) < 1e-12  # theta (azimuth)
    assert abs(sph[2] - np.pi / 2) < 1e-12  # phi (pole)


def test_sph2cart_roundtrip():
    rng = np.random.default_rng(123)
    cart = rng.standard_normal((50, 3))
    recovered = sph2cart(cart2sph(cart))
    np.testing.assert_allclose(recovered, cart, atol=1e-12)


def test_geo2sph_sph2geo_roundtrip():
    geo = np.array([[6371.0, 45.0, 30.0], [6371.0, -120.0, -45.0]])
    sph = geo2sph(geo, degrees=True)
    recovered = sph2geo(sph, degrees=True)
    np.testing.assert_allclose(recovered, geo, atol=1e-12)


def test_cart2polar_basic():
    cart = np.array([1.0, 0.0])
    polar = cart2polar(cart)
    assert abs(polar[0] - 1.0) < 1e-12
    assert abs(polar[1] - 0.0) < 1e-12


def test_degrees_flag():
    cart = np.array([0.0, 1.0, 0.0])
    sph_rad = cart2sph(cart, degrees=False)
    sph_deg = cart2sph(cart, degrees=True)
    np.testing.assert_allclose(np.rad2deg(sph_rad[1:]), sph_deg[1:], atol=1e-10)


def test_batch_shapes():
    cart = np.ones((10, 3))
    sph = cart2sph(cart)
    assert sph.shape == (10, 3)
