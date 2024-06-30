import pandas as pd
import numpy as np

# Generate sample collar data
collar_data = {
    'borehole_id': [f'CPT{i}' for i in range(1, 13)],
    'X': np.random.uniform(1000, 1100, 12),
    'Y': np.random.uniform(2000, 2100, 12),
    'Z': np.random.uniform(150, 200, 12)
}

collar_df = pd.DataFrame(collar_data)
collar_df.to_csv('collar2.csv', index=False)

# Generate sample lithology data
Np = 200
depths = np.linspace(0, 100, Np)  # 2000 data points per sounding
lithology_data = []

# Define soil behavior types
SBT_types = list(range(1, 10))

for sounding_id in collar_data['borehole_id']:
    Z_top = collar_df[collar_df['borehole_id'] == sounding_id]['Z'].values[0]
    
    # Sinusoidal baseline with random noise
    penetration_resistance = 15 + 10 * np.sin(2 * np.pi * depths / 100) + np.random.uniform(-5, 5, Np)
    friction_ratio = 1 + 0.5 * np.sin(4 * np.pi * depths / 100) + np.random.uniform(-0.1, 0.1, Np)
    
    SBT_index = np.random.choice(SBT_types, Np)  # Random categorical SBT index
    
    for i in range(Np):
        depth = depths[i]
        elevation = Z_top - depth
        lithology_data.append({
            'borehole_id': sounding_id,
            'depth': depth,
            'elevation': elevation,
            'penetration_resistance': penetration_resistance[i],
            'friction_ratio': friction_ratio[i],
            'SBT_index': SBT_index[i]
        })

lithology_df = pd.DataFrame(lithology_data)
lithology_df.to_csv('lithology2.csv', index=False)
