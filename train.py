# Parametry
import numpy as np

num_samples = 8192
decimal_places = 8
range_min, range_max = -8, 8

# Generování dat
x_values = np.random.uniform(range_min, range_max, num_samples)
y_values = np.random.uniform(range_min, range_max, num_samples)
z_values_smoother = 4 * np.sin(0.5 * x_values) * np.cos(0.5 * y_values) + 0.5 * np.sin(0.5 * x_values)

# Zaokrouhlení
data_smoother_waves = np.column_stack((x_values, y_values, z_values_smoother))
data_smoother_waves = np.round(data_smoother_waves, decimals=decimal_places)

# Uložení do souboru
output_file = 'train/vlny.txt'
np.savetxt(output_file, data_smoother_waves, fmt=f'%.{decimal_places}f')

output_file

import numpy as np

def generate_mexican_hat_data(num_samples=8192, decimal_places=8, range_min=-8, range_max=8, output_file='train/mexican_hat.txt'):
    # Generování dat
    x_values = np.random.uniform(range_min, range_max, num_samples)
    y_values = np.random.uniform(range_min, range_max, num_samples)
    r = np.sqrt(x_values**2 + y_values**2)
    z_values = np.sin(r) / r
    
    # Odstranění nulových hodnot
    mask = (x_values != 0) & (y_values != 0) & (z_values != 0)
    x_values, y_values, z_values = x_values[mask], y_values[mask], z_values[mask]
    
    # Omezení na požadovaný počet příkladů
    x_values, y_values, z_values = x_values[:num_samples], y_values[:num_samples], z_values[:num_samples]
    
    # Zaokrouhlení
    data = np.column_stack((x_values, y_values, z_values))
    data = np.round(data, decimals=decimal_places)
    
    # Uložení do souboru
    np.savetxt(output_file, data, fmt=f'%.{decimal_places}f')
    print(f'Data uložena do {output_file}')

# Generování souboru
if __name__ == "__main__":
    generate_mexican_hat_data()


import numpy as np

# Parametry
num_samples = 4096
range_min, range_max = -8, 8
radius = 8

# Generování dat
data = []
while len(data) < num_samples:
    x = np.random.uniform(range_min, range_max)
    y = np.random.uniform(range_min, range_max)

    # Vynechání nulových hodnot
    if x == 0 or y == 0:
        continue

    # Výpočet z pro horní polokouli
    z_squared = radius**2 - x**2 - y**2
    if z_squared > 0:
        z = np.sqrt(z_squared)
        data.append((x, y, z))

# Uložení do souboru
output_path = 'train/sphere.txt'
np.savetxt(output_path, data, fmt='%.10f')

output_path

