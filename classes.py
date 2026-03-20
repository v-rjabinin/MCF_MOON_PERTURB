from typing import Union, Any, Optional
from dataclasses import dataclass

import numpy as np
import warnings
import data


Real = Union[int, float, np.integer, np.floating]
Float = Union[float, np.floating]
Int = Union[int, np.integer]


@dataclass
class Orbit:
    a: Real                    # Semimajor axis (a): kilometers (km)
    e: Real                    # Eccentricity (e): dimensionless (-)
    i: Real                    # Inclination (i): radians (rad)
    u: np.ndarray              # Argument of latitude (u): radians (rad)
    omega: Real                # Longitude of the ascending node (Ω): radians (rad)
    arg_per: Real              # Argument of periapsis/perigee (ω): radians (rad)

    def __post_init__(self):
        self._validate_attr(self.a, "a", min_lim=0, max_lim=None, min_exclusive=True, max_exclusive=False)
        self._validate_attr(self.e, "e", min_lim=0, max_lim=1, min_exclusive=False, max_exclusive=True)
        self._validate_attr(self.i, "i", min_lim=0, max_lim=np.pi, min_exclusive=False, max_exclusive=False)
        self._validate_attr(self.omega, "omega", min_lim=0, max_lim=2 * np.pi, min_exclusive=False, max_exclusive=False)
        self._validate_attr(self.arg_per, "arg_per", min_lim=0, max_lim=2 * np.pi, min_exclusive=False, max_exclusive=False)

        if isinstance(self.u, np.ndarray):
            if self.u.ndim > 1:
                warnings.warn(f"Expected 1D array, got {self.u.ndim}D. Input will be flattened", UserWarning, stacklevel=2)
                self.u = self.u.ravel()

            if not (np.issubdtype(self.u.dtype, np.floating) or np.issubdtype(self.u.dtype, np.integer)):
                raise TypeError(f"Attribute 'u' must have the dtype that is a subdtype of numpy.integer or numpy.floating")

            for i in range(len(self.u)):
                self._validate_value(self.u[i], f"u[{i}]", min_lim=0, max_lim=2 * np.pi, min_exclusive=False, max_exclusive=False)

    def _validate_attr(self, value: Any, arg_name: str, min_lim: Optional[Real], max_lim: Optional[Real], min_exclusive: bool = False, max_exclusive: bool = False) -> None:
        self._validate_type(value, arg_name)
        self._validate_value(value, arg_name, min_lim, max_lim, min_exclusive, max_exclusive)

    @staticmethod
    def _validate_type(value: Any, arg_name: str) -> None:
        if not isinstance(value, (np.integer, np.floating, int, float)):
            raise TypeError(f"Attribute '{arg_name}' must be a real number")

    @staticmethod
    def _validate_value(value: Real, arg_name: str, min_lim: Optional[Real], max_lim: Optional[Real], min_exclusive: bool = False, max_exclusive: bool = False) -> None:
        if min_lim is None:
            check = value >= max_lim if max_exclusive else value > max_lim
            sym = ")" if max_exclusive else "]"

            if check:
                raise ValueError(f"Attribute '{arg_name}' must be in (-∞; {max_lim}{sym}")
        elif max_lim is None:
            check = value <= min_lim if min_exclusive else value < min_lim
            sym = "(" if min_exclusive else "["

            if check:
                raise ValueError(f"Attribute '{arg_name}' must be in {sym}{min_lim}; +∞)")
        else:
            check_2 = value >= max_lim if max_exclusive else value > max_lim
            sym_2 = ")" if max_exclusive else "]"

            check_1 = value <= min_lim if min_exclusive else value < min_lim
            sym_1 = "(" if min_exclusive else "["

            if check_1 or check_2:
                raise ValueError(f"Attribute '{arg_name}' must be in {sym_1}{min_lim}; {max_lim}{sym_2}")


class CoordValidator:
    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __set__(self, instance, value):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Parameter '{self.name}' must be a numpy.ndarray")
        elif not (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.integer)):
            raise TypeError(f"Parameter '{self.name}' must have the dtype that is a subdtype of numpy.integer or numpy.floating")
        elif not (value.ndim == 2 and value.shape[0] == 3):
            raise ValueError(f"Parameter '{self.name}' must be 2D array with the first axis equal three, i. e. (3, n)")

        setattr(instance, self.name, value)

    def __get__(self, instance, owner):
        return getattr(instance, self.name, None)


class CoordTransformer:
    eci_coord = CoordValidator()
    own_gocs_coord = CoordValidator()
    extern_gocs_coord = CoordValidator()

    def __init__(self, orbit: Orbit):
        self.orbit = orbit

        self._eci_coord = None
        self._own_gocs_coord = None
        self._extern_gocs_coord = None

        self.__check_angles = None

    def transform_to_gocs(self) -> None:
        # GOCS - Geocentric Orbital Coordinate System

        u, arg_per = self.orbit.u, self.orbit.arg_per
        a, e = self.orbit.a, self.orbit.e

        nu = u - arg_per
        p = a * (1 - e * e)
        r = p / (1 + e * np.cos(nu))

        coord = np.zeros(shape=(3, u.shape[0]), dtype=np.float64)

        coord[0, :] = np.cos(nu) * r
        coord[1, :] = np.sin(nu) * r

        self.own_gocs_coord = coord

    def transform_to_eci(self) -> None:
        # ECI - Earth-centered inertial coordinate system

        u, arg_per = self.orbit.u, self.orbit.arg_per
        omega, i = self.orbit.omega, self.orbit.i
        a, e = self.orbit.a, self.orbit.e

        r_a = a * (1 - e ** 2) / (1 + e * np.cos(u - arg_per))

        x_a = r_a * (np.cos(u) * np.cos(omega) - np.sin(u) * np.sin(omega) * np.cos(i))
        y_a = r_a * (np.cos(u) * np.sin(omega) + np.sin(u) * np.cos(omega) * np.cos(i))
        z_a = r_a * np.sin(u) * np.sin(i)

        self.eci_coord = np.vstack([x_a, y_a, z_a])

    def transform_to_external_gocs(self, omega: Real, i: Real, arg_per: Real) -> None:
        # GOCS - Geocentric Orbital Coordinate System of the external body

        if self.eci_coord is None:
            raise AttributeError("Before running this method you must run 'transform_to_eci' method")

        self.__check_angles = np.array([omega, i, arg_per], dtype=np.float64)

        rotation_1 = [[np.cos(omega), np.sin(omega), 0], [-np.sin(omega), np.cos(omega), 0], [0, 0, 1]]
        rotation_2 = [[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]]
        rotation_3 = [[np.cos(arg_per), np.sin(arg_per), 0], [-np.sin(arg_per), np.cos(arg_per), 0], [0, 0, 1]]

        r_1 = np.array(rotation_1, dtype=np.float64)
        r_2 = np.array(rotation_2, dtype=np.float64)
        r_3 = np.array(rotation_3, dtype=np.float64)

        self.extern_gocs_coord = r_3 @ (r_2 @ (r_1 @ self.eci_coord))

    def get_rotation_angles(self):
        return self.__check_angles


class MoonAccelerComputer:
    mu_moon = data.mu_moon

    def __init__(self, sat_coord: CoordTransformer, moon_coord: CoordTransformer, orbit: Orbit):
        self.moon_coord = moon_coord
        self.sat_coord = sat_coord
        self.orbit = orbit

        curr_angles = np.array([orbit.omega, orbit.i, orbit.arg_per], dtype=np.float64)
        moon_angles = moon_coord.get_rotation_angles()

        if moon_angles is None or np.any(curr_angles != moon_coord.get_rotation_angles()):
            warnings.warn(f"The last angles of rotation of the axes for the Moon differ from the corresponding angles of the satellite's orbit.", UserWarning, stacklevel=2)

    def __call__(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        mu_moon = self.mu_moon

        sat_coord = self.sat_coord.own_gocs_coord
        moon_coord = self.moon_coord.extern_gocs_coord

        r_1 = sat_coord                 # Earth-Satellite Vector
        r_12 = -moon_coord              # Moon-Earth Vector
        r_2 = r_1 + r_12                # Moon-Satellite Vector

        coef_1, coef_2 = -mu_moon / (np.sqrt(np.sum(r_2 ** 2, axis=0)) ** 3), mu_moon / (np.sqrt(np.sum(r_12 ** 2, axis=0)) ** 3)
        F = coef_1 * r_2 + coef_2 * r_12

        nu, acceleration = self.orbit.u - self.orbit.arg_per, np.empty_like(F, dtype=np.float64)

        acceleration[0, :] = F[0, :] * np.cos(nu) + F[1, :] * np.sin(nu)
        acceleration[1, :] = -F[0, :] * np.sin(nu) + F[1, :] * np.cos(nu)
        acceleration[2, :] = F[2, :]

        return acceleration, np.sqrt(np.sum(r_2 ** 2, axis=0))

    @property
    def orbit(self):
        return self._orbit

    @orbit.setter
    def orbit(self, orbit):
        if not isinstance(orbit, Orbit):
            raise TypeError("Parameter 'orbit' must be an Orbit object")

        self._orbit = orbit

    @property
    def sat_coord(self):
        return self._sat_coord

    @sat_coord.setter
    def sat_coord(self, sat_coord):
        if not isinstance(sat_coord, CoordTransformer):
            raise TypeError("Parameter 'sat_coord' must be a CoordTransformer object")
        elif sat_coord.own_gocs_coord is None:
            raise AttributeError("Parameter 'cat_coord' must have 'own_gocs_coord' attribute set")

        self._sat_coord = sat_coord

    @property
    def moon_coord(self):
        return self._moon_coord

    @moon_coord.setter
    def moon_coord(self, moon_coord):
        if not isinstance(moon_coord, CoordTransformer):
            raise TypeError("Parameter 'moon_coord' must be a CoordTransformer object")
        elif moon_coord.extern_gocs_coord is None:
            raise AttributeError("Parameter 'moon_coord' must have 'extern_gocs_coord' attribute set")

        self._moon_coord = moon_coord