# Parametry
import numpy as np

num_samples = 12000
decimal_places = 8
range_min, range_max = -8, 8

# Generování dat
x_values = np.random.uniform(range_min, range_max, num_samples)
y_values = np.random.uniform(range_min, range_max, num_samples)
z_values_smoother = 4 * np.sin(0.4 * x_values) * np.cos(0.4 * y_values) + 0.4 * np.sin(0.4 * x_values)

# Zaokrouhlení
data_smoother_waves = np.column_stack((x_values, y_values, z_values_smoother))
data_smoother_waves = np.round(data_smoother_waves, decimals=decimal_places)

# Uložení do souboru
output_file = 'train/vlny.txt'
np.savetxt(output_file, data_smoother_waves, fmt=f'%.{decimal_places}f')

output_file

import numpy as np

# Parametry generování dat
num_samples = 1000  # Počet příkladů
x_range = (-2.83, 2.83)  # Upravený rozsah pro x
y_range = (-2.83, 2.83)  # Upravený rozsah pro y
a = 1.0
a = 1.0
b = 1.0

# Generování náhodných hodnot pro x a y
x_values = np.random.uniform(x_range[0], x_range[1], num_samples)
y_values = np.random.uniform(y_range[0], y_range[1], num_samples)

# Výpočet hodnot z podle rovnice hyperbolického sedla
z_values = (x_values ** 2) / (a ** 2) - (y_values ** 2) / (b ** 2)

# Uložení do souboru
output_file = "train\saddle.txt"
with open(output_file, "w") as f:
    for x, y, z in zip(x_values, y_values, z_values):
        f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")

print(f"{num_samples} příkladů bylo vygenerováno a uloženo do '{output_file}'.")

import numpy as np

# Parametry generování dat
num_samples = 4000  # Počet příkladů
radius = 8.0  # Poloměr polokoule

# Generování náhodných hodnot pro x a y uvnitř kruhu s daným poloměrem
x_values = []
y_values = []
z_values = []

for _ in range(num_samples):
    while True:
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        if x**2 + y**2 <= radius**2:
            z = np.sqrt(radius**2 - x**2 - y**2)
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)
            break

# Uložení do souboru
output_file = "train\sphere.txt"
with open(output_file, "w") as f:
    for x, y, z in zip(x_values, y_values, z_values):
        f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")

print(f"{num_samples} příkladů bylo vygenerováno a uloženo do '{output_file}'.")