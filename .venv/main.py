import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support

# Paramètres de la simulation
np.random.seed(0)  # Pour la reproductibilité
num_particles = 100000
g = -9.81  # Accélération due à la gravité (m/s²)
rho = 1.225  # Densité de l'air (kg/m³)
dt = 0.01  # Intervalle de temps (s)

# Paramètres gaussiens
v0_mean = 10  # Vitesse initiale moyenne (m/s)
v0_std = 2    # Écart-type de la vitesse initiale
m_mean = 0.01  # Masse moyenne de la particule (kg)
m_std = 0.002  # Écart-type de la masse
A_mean = np.pi * (0.005 ** 2)  # Section efficace moyenne (m²)
A_std = np.pi * (0.001 ** 2)  # Écart-type de la section efficace
Cd_mean = 0.47  # Coefficient de traînée moyen
Cd_std = 0.05   # Écart-type du coefficient de traînée

# Génération des paramètres selon une distribution gaussienne
v0_values = np.random.normal(v0_mean, v0_std, num_particles)
m_values = np.random.normal(m_mean, m_std, num_particles)
A_values = np.random.normal(A_mean, A_std, num_particles)
Cd_values = np.random.normal(Cd_mean, Cd_std, num_particles)

# Fonction pour calculer la portée de projection
def calculate_range(params):
    v0, m, A, Cd = params
    theta = np.radians(30)  # Angle d'éjection (radians)
    vx, vy = v0 * np.cos(theta), v0 * np.sin(theta)
    x, y = 0.0, 0.0

    while y >= 0:
        # Calcul des forces
        v = np.sqrt(vx**2 + vy**2)
        Fd_x = -0.5 * rho * Cd * A * vx * v
        Fd_y = -0.5 * rho * Cd * A * vy * v

        # Mise à jour des vitesses
        vx += (Fd_x / m) * dt
        vy += (g + Fd_y / m) * dt

        # Mise à jour des positions
        x += vx * dt
        y += vy * dt

    return x

def main():
    # Préparation des paramètres pour multiprocessing
    params = list(zip(v0_values, m_values, A_values, Cd_values))

    # Utilisation de multiprocessing pour calculer les portées
    with Pool() as pool:
        ranges = np.array(pool.map(calculate_range, params))

    # Statistiques
    mean_range = np.mean(ranges)
    std_range = np.std(ranges)

    # Tracé de la distribution des portées
    plt.hist(ranges, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(mean_range, color='red', linestyle='dashed', linewidth=1, label=f'Moyenne: {mean_range:.2f} m')
    plt.axvline(mean_range + std_range, color='green', linestyle='dashed', linewidth=1, label=f'+1 Écart-type: {mean_range + std_range:.2f} m')
    plt.axvline(mean_range - std_range, color='green', linestyle='dashed', linewidth=1, label=f'-1 Écart-type: {mean_range - std_range:.2f} m')
    plt.xlabel('Portée de projection (m)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des portées de projection')
    plt.legend()
    plt.grid(True)
    plt.show()

    return mean_range, std_range

if __name__ == '__main__':
    freeze_support()
    mean_range, std_range = main()
