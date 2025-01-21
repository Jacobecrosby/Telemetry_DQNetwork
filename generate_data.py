import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 5.972e24  # Earth's mass (kg)
R = 6371e3  # Earth's radius (m)
J2 = 1.08263e-3  # Earth's oblateness factor
mu = G * M  # Earth's gravitational parameter (m^3/s^2)
rho0 = 1.225  # Sea level atmospheric density (kg/m^3)
H = 8500  # Scale height for atmosphere (m)
Cd = 2.2  # Drag coefficient
A = 4  # Cross-sectional area (m^2)
m = 500  # Satellite mass (kg)

# Time settings
time_step = 10  # seconds
total_steps = 10000  # Total time steps

# Functions for perturbations
def j2_perturbation(a, i, RAAN):
    # J2 perturbation for inclination and RAAN drift
    RAAN_dot = -3 / 2 * np.sqrt(mu) * J2 * (R**2 / a**3.5) * np.cos(i)
    i_dot = 0  # For simplicity, assuming no significant inclination change
    return RAAN_dot, i_dot

def atmospheric_drag(altitude, velocity):
    # Drag acceleration (simple exponential atmosphere model)
    rho = rho0 * np.exp(-(altitude - R) / H)
    drag_acc = 0.5 * rho * Cd * A / m * velocity**2
    return drag_acc

def solar_radiation_pressure():
    # Simplified solar radiation force (random fluctuation for now)
    return np.random.normal(0, 1e-6)

# Generate telemetry data
def generate_telemetry(time_steps, orbit_type="LEO"):
    """
    Generate telemetry data for either Low Earth Orbit (LEO) or Geostationary Orbit (GEO).
    
    Parameters:
        time_steps (int): Number of time steps to simulate.
        orbit_type (str): "LEO" for Low Earth Orbit, "GEO" for Geostationary Orbit.
    
    Returns:
        pd.DataFrame: Simulated telemetry data.
    """
    if orbit_type == "LEO":
        a = R + 500e3  # Semi-major axis for LEO (500 km altitude)
        i = np.radians(45)  # Inclination
    elif orbit_type == "GEO":
        a = R + 35786e3  # Semi-major axis for GEO (35,786 km altitude)
        i = np.radians(0)  # Near-equatorial orbit
    else:
        raise ValueError("Invalid orbit type. Choose 'LEO' or 'GEO'.")

    ecc = 0.01  # Eccentricity
    RAAN = np.radians(60)  # Right Ascension of Ascending Node
    omega = np.radians(30)  # Argument of periapsis
    telemetry = []
    RAAN_current = RAAN
    i_current = i

    for t in range(time_steps):
        # Simplified position and velocity
        theta = 2 * np.pi * (t * time_step) / (2 * np.pi * np.sqrt(a**3 / mu))  # True anomaly
        altitude = a
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        z = a * np.sin(i_current) * np.sin(theta)

        vx = -np.sin(theta) * np.sqrt(mu / a)
        vy = np.cos(theta) * np.sqrt(mu / a)
        vz = 0

        # Apply perturbations
        RAAN_dot, i_dot = j2_perturbation(a, i_current, RAAN_current)
        drag_acc = atmospheric_drag(altitude, np.sqrt(vx**2 + vy**2 + vz**2))
        solar_pressure = solar_radiation_pressure()

        # Update RAAN and inclination
        RAAN_current += RAAN_dot * time_step
        i_current += i_dot * time_step

        # Apply drag to velocity
        vx -= drag_acc * vx * time_step
        vy -= drag_acc * vy * time_step

        telemetry.append({
            "timestamp": t * time_step,
            "x": x + np.random.normal(0, 100),
            "y": y + np.random.normal(0, 100),
            "z": z + np.random.normal(0, 50),
            "vx": vx + np.random.normal(0, 0.01),
            "vy": vy + np.random.normal(0, 0.01),
            "vz": vz + np.random.normal(0, 0.01),
            "RAAN": RAAN_current,
            "inclination": i_current,
            "temperature": 20 + 5 * np.sin(2 * np.pi * t / total_steps) + np.random.normal(0, 1),
            "battery": 100 - t * 0.01
        })

    return pd.DataFrame(telemetry)


def plot_data(telemetry_data, orbit):
    # Visualization: Orbit in X-Y Plane
    plt.figure(figsize=(10, 6))
    plt.plot(telemetry_data["x"], telemetry_data["y"], label="Orbit (x-y plane)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Satellite Orbit in X-Y Plane")
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/orbit_xy_plane_{orbit}.png", dpi=300)
    plt.close()

    # Visualization: Battery Level Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(telemetry_data["timestamp"], telemetry_data["battery"], label="Battery Level")
    plt.xlabel("Time (s)")
    plt.ylabel("Battery Level (%)")
    plt.title("Satellite Battery Level Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/battery_level_{orbit}.png", dpi=300)
    plt.close()

    # Visualization: Orbit in Y-Z Plane
    plt.figure(figsize=(10, 6))
    plt.plot(telemetry_data["y"], telemetry_data["z"], label="Orbit (y-z plane)")
    plt.xlabel("Y Position (m)")
    plt.ylabel("Z Position (m)")
    plt.title("Satellite Orbit in Y-Z Plane")
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/orbit_yz_plane_{orbit}.png", dpi=300)
    plt.close()

    # Generate and save telemetry data for LEO and GEO
for orbit in ["LEO", "GEO"]:
    telemetry_data = generate_telemetry(total_steps, orbit_type=orbit)
    plot_data(telemetry_data, orbit)
    telemetry_data.to_csv(f"data/satellite_telemetry_{orbit.lower()}.csv", index=False)
    print(f"{orbit} telemetry data saved.")