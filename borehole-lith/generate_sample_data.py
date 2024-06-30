import pandas as pd
import numpy as np

# Generate sample collar data
collar_data = {
    'borehole_id': [f'BH{i}' for i in range(1, 13)],
    'X': np.random.uniform(1000, 1100, 12),
    'Y': np.random.uniform(2000, 2100, 12),
    'Z': np.random.uniform(150, 200, 12)
}

collar_df = pd.DataFrame(collar_data)
collar_df.to_csv('collar.csv', index=False)

# Generate sample lithology data
lithologies = ['Sand', 'Clay', 'Gravel', 'Silt', 'Limestone']
lithology_data = []

for bh in collar_data['borehole_id']:
    depth_intervals = sorted(np.random.uniform(0, 100, 10))
    for i in range(len(depth_intervals) - 1):
        start_depth = depth_intervals[i]
        end_depth = depth_intervals[i+1]
        start_elevation = collar_df[collar_df['borehole_id'] == bh]['Z'].values[0] - start_depth
        end_elevation = collar_df[collar_df['borehole_id'] == bh]['Z'].values[0] - end_depth
        lithology_data.append({
            'borehole_id': bh,
            'start_depth': start_depth,
            'end_depth': end_depth,
            'start_elevation': start_elevation,
            'end_elevation': end_elevation,
            'lithology': np.random.choice(lithologies)
        })

lithology_df = pd.DataFrame(lithology_data)
lithology_df.to_csv('lithology.csv', index=False)
