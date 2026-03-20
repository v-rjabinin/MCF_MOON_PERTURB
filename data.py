import numpy as np

# Initial data (Variant №18)

# For the spacecraft being considered
i = 32.4                    # Inclination (deg, °)
h_a = 164_000               # Apocenter height (km)
h_p = 125_000               # Pericenter height (km)
omega = 40                  # Longitude of the ascending node (deg, °)
u = 10                      # Argument of latitude (deg, °)
s_a = 12                    # Satellite cross-section (sq. m)
m = 1200                    # Mass (kg)
c_xa = 3.5                  # Aerodynamic drag coefficient (no dim)
F_a = 100                   # Solar radiation flux (W / sq m)
arg_per = 0                 # Argument of periapsis, (rad)
tao = -12401.300834991971   # Time of periapsis passage, (s)

# For the Moon
i_moon = 5.145              # Inclination (deg, °)
h_a_moon = 405_696          # Apocenter height (km)
h_p_moon = 363_104          # Pericenter height (km)
omega_moon = 30             # Longitude of the ascending node (deg, °)
u_moon = 5                  # Argument of latitude (deg, °)
arg_per_moon = 0            # Argument of periapsis, (rad)
tao_moon = -30233.628764780 # Time of periapsis passage, (s)

# Core constants
R = 6371                    # Radius of the Earth (km)
mu = 398_600.4418           # Standard gravitational parameter of the Earth (km^3 / s^2)
omega_earth = 7.292115e-5   # Earth's angular velocity (rad / s)
mu_moon = 4902.8            # Standard gravitational parameter of the Moon (km^3 / s^2)

# Change of measurement units (deg -> rad)
i = np.radians(i)
omega = np.radians(omega)
u = np.radians(u)

i_moon = np.radians(i_moon)
omega_moon = np.radians(omega_moon)
u_moon = np.radians(u_moon)


# Calculation of necessary additional values
r_a = h_a + R
r_p = h_p + R
a = (r_a + r_p) / 2                                 # Semi-major axis of the satellite orbit (km)
e = (r_a - r_p) / (2 * a)                           # Eccentricity of the satellite orbit
p = a * (1 - e * e)                                 # Focal parameter, (km)
T = 2 * np.pi * np.sqrt((a ** 3) / mu)              # Orbital period of the satellite, (s)
n = 2 * np.pi / T                                   # Mean motion of the satellite, (s^(-1))

r_a_moon = h_a_moon + R
r_p_moon = h_p_moon + R
a_moon = (r_a_moon + r_p_moon) / 2                  # Semi-major axis of the Moon orbit (km)
e_moon = (r_a_moon - r_p_moon) / (2 * a_moon)       # Eccentricity of the Moon orbit
T_moon = 2 * np.pi * np.sqrt((a_moon ** 3) / mu)    # Orbital period of the Moon, (s)
n_moon = 2 * np.pi / T_moon                         # Mean motion of the Moon, (s^(-1))

t_start = 0.0