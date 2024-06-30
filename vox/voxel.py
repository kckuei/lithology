import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
collar_df = pd.read_csv(r'..\borehole-lith\collar.csv')
lithology_df = pd.read_csv(r'..\borehole-lith\lithology.csv')

# Convert lithology to numerical codes
lithology_df['lithology_code'] = pd.Categorical(lithology_df['lithology']).codes

# Define the grid dimensions
nx, ny, nz = 50, 50, 50  # Number of divisions in each dimension
x = np.linspace(collar_df['X'].min(), collar_df['X'].max(), nx)
y = np.linspace(collar_df['Y'].min(), collar_df['Y'].max(), ny)
z = np.linspace(collar_df['Z'].min(), collar_df['Z'].max(), nz)
grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

# Prepare the data for interpolation
mid_elevations = (lithology_df['start_elevation'] + lithology_df['end_elevation']) / 2
points = np.vstack((collar_df.set_index('borehole_id').loc[lithology_df['borehole_id'], 'X'].values,
                    collar_df.set_index('borehole_id').loc[lithology_df['borehole_id'], 'Y'].values,
                    mid_elevations)).T
values = lithology_df['lithology_code'].values  # Use numerical codes for interpolation

# Interpolate using griddata
grid_values = griddata(points, values, (grid_x, grid_y, grid_z), method='nearest')

# Convert interpolated values to integers
grid_values = grid_values.astype(int)

# Handle NaN values by filling with the nearest values
grid_values = np.nan_to_num(grid_values, nan=np.nanmean(values)).astype(int)

# Create a boolean array for voxel plotting
voxelarray = grid_values >= 0  # want to plot all of them for now
# voxelarray = grid_values == 3  

# # Create a color map for visualization
# norm = plt.Normalize(grid_values.min(), grid_values.max())
# colors = plt.cm.tab10(norm(grid_values % 10))

# # Mask out the transparent areas (where voxelarray is False)
# colors = np.where(voxelarray[..., None], colors, 0)




# Create a consistent color mapping for visualization
unique_codes = np.unique(lithology_df['lithology_code'])
color_map = plt.cm.tab10
colors = np.zeros(grid_values.shape + (4,))
for code in unique_codes:
    colors[grid_values == code] = color_map(code / len(unique_codes))






# Plot the voxel representation
fig = plt.figure(figsize=(20, 10))

# 3D Voxel Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.voxels(voxelarray, facecolors=colors, edgecolor='k')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Add a legend
handles = [plt.Line2D([0], [0], color=color_map(i / len(unique_codes)), lw=4) for i in unique_codes]
labels = pd.Categorical(lithology_df['lithology']).categories
ax1.legend(handles, labels, loc='best', title="Lithology")

# Plot a 2D Slice
slice_index = 25  # Change this index to get different slices
slice_data = grid_values[:, :, slice_index]
slice_colors = colors[:, :, slice_index]

ax2 = fig.add_subplot(122)
ax2.imshow(slice_colors, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(f'Z Slice at index {slice_index}')

plt.show()


# Plot all the slices (horizontal)
for i in range(50):
    slice_index = 25  # Change this index to get different slices
    slice_data = grid_values[:, :, i]
    slice_colors = colors[:, :, i]
    plt.imshow(slice_colors, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
    plt.show()
    
# Plot all the slices (vertical)
for i in range(50):
    slice_index = 25  # Change this index to get different slices
    slice_data = grid_values[:, i, :]
    slice_colors = colors[:, i, :]
    plt.imshow(slice_colors, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
    plt.show()