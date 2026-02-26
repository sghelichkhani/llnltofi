import numpy as np
from numpy.typing import ArrayLike, NDArray


def cart2sph(cartesian_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """Take shape (N,3) or (3,) cartesian coord_array and return an array of the same
    shape in spherical polar form (r, theta, phi).

    Use radians for angles by default, degrees if ``degrees == True``."""

    cartesian_coord_array = np.array(cartesian_coord_array, dtype="float64")
    spherical_coord_array = np.empty(cartesian_coord_array.shape)

    spherical_coord_array[..., 0] = np.linalg.norm(cartesian_coord_array, axis=-1)
    spherical_coord_array[..., 1] = np.arctan2(
        cartesian_coord_array[..., 1], cartesian_coord_array[..., 0]
    )
    spherical_coord_array[..., 2] = np.arccos(
        cartesian_coord_array[..., 2] / spherical_coord_array[..., 0]
    )

    if degrees:
        spherical_coord_array[..., 1:] = np.rad2deg(spherical_coord_array[..., 1:])

    return spherical_coord_array


def sph2cart(spherical_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """Take shape (N,3) or (3,) spherical_coord_array (radius, azimuth, pole) and
    return an array of the same shape in cartesian coordinate form (x, y, z).

    Use radians for angles by default, degrees if ``degrees == True``."""

    spherical_coord_array = np.array(spherical_coord_array, dtype="float64")
    cartesian_coord_array = np.empty(spherical_coord_array.shape)

    if degrees:
        spherical_coord_array[..., 1:] = np.deg2rad(spherical_coord_array[..., 1:])

    cartesian_coord_array[..., 0] = (
        spherical_coord_array[..., 0]
        * np.cos(spherical_coord_array[..., 1])
        * np.sin(spherical_coord_array[..., 2])
    )
    cartesian_coord_array[..., 1] = (
        spherical_coord_array[..., 0]
        * np.sin(spherical_coord_array[..., 1])
        * np.sin(spherical_coord_array[..., 2])
    )
    cartesian_coord_array[..., 2] = spherical_coord_array[..., 0] * np.cos(
        spherical_coord_array[..., 2]
    )

    return cartesian_coord_array


def geo2sph(geographical_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """Take shape (N,2), (N,3), (2,), or (3,) geographical_coord_array
    ([radius], lon, lat) and return an array of the same shape in spherical
    coordinate form ([radius], azimuth, pole)."""

    geographical_coord_array = np.array(geographical_coord_array, dtype="float64")
    spherical_coord_array = geographical_coord_array.copy()

    spherical_coord_array[..., -1] = 90 - spherical_coord_array[..., -1]

    if not degrees:
        spherical_coord_array[..., -2:] = np.deg2rad(spherical_coord_array[..., -2:])

    return spherical_coord_array


def sph2geo(spherical_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """Take shape (N,2), (N,3), (2,), or (3,) spherical_coord_array
    ([radius], azimuth, pole) and return an array of the same shape in
    geographical coordinate form ([radius], lon, lat)."""

    spherical_coord_array = np.array(spherical_coord_array, dtype="float64")
    geographical_coord_array = spherical_coord_array.copy()

    if not degrees:
        geographical_coord_array[..., -2:] = np.rad2deg(
            geographical_coord_array[..., -2:]
        )

    geographical_coord_array[..., -1] = 90 - geographical_coord_array[..., -1]

    return geographical_coord_array


def cart2polar(cartesian_coord_array: ArrayLike, degrees: bool = False) -> NDArray:
    """Take shape (N,2) cartesian_coord_array and return an array of the same shape
    in polar coordinates (radius, azimuth).

    Use radians for angles by default, degrees if ``degrees == True``."""

    cartesian_coord_array = np.array(cartesian_coord_array, dtype="float64")
    polar_coord_array = np.empty(cartesian_coord_array.shape)

    polar_coord_array[..., 0] = np.linalg.norm(cartesian_coord_array, axis=-1)
    polar_coord_array[..., 1] = np.arctan2(
        cartesian_coord_array[..., 1], cartesian_coord_array[..., 0]
    )

    if degrees:
        polar_coord_array[..., 1] = np.rad2deg(polar_coord_array[..., 1])

    return polar_coord_array


def great_circle_distance(
    array_1: ArrayLike,
    array_2: ArrayLike,
    coordinate_system: str = "spherical",
    sphere_radius: bool | float = False,
) -> float:
    """Calculate the haversine-based distance between two arrays of points on the
    surface of a sphere (array shape must be (N,3) or (3,)).

    ``coordinate_system`` can be 'spherical' or 'cartesian'.  If ``sphere_radius``
    is not given, the radius from array_1 is used."""

    assert coordinate_system in ["spherical", "cartesian"]
    array_1, array_2 = np.array(array_1), np.array(array_2)

    if coordinate_system == "cartesian":
        array_1 = cart2sph(array_1)
        array_2 = cart2sph(array_2)

    if not sphere_radius:
        sphere_radius = array_1[..., 0]

    phi_1 = array_1[..., 1]
    phi_2 = array_2[..., 1]
    theta_1 = array_1[..., 2]
    theta_2 = array_2[..., 2]

    spherical_distance = (
        2.0
        * sphere_radius
        * np.arcsin(
            np.sqrt(
                ((1 - np.cos(theta_2 - theta_1)) / 2.0)
                + np.sin(theta_1)
                * np.sin(theta_2)
                * ((1 - np.cos(phi_2 - phi_1)) / 2.0)
            )
        )
    )

    return spherical_distance
