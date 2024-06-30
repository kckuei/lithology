import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# Load the data
# collar_df = pd.read_csv('collar.csv')
# lithology_df = pd.read_csv('lithology.csv')
collar_df = pd.read_csv('collar2.csv')
lithology_df = pd.read_csv('lithology2.csv')

# Define colors for SBT indices using a colormap
# SBT_colors = plt.cm.get_cmap('tab10', 9)  # Using a colormap with 9 distinct colors
norm = plt.Normalize(1,9)
SBT_colors = LinearSegmentedColormap.from_list("", ['#D3291C', '#B36A3F', '#4A5777', '#479085', '#7EC4A0', '#BDA464', '#EB9D4A', '#999999', '#DEDEDE'], N=9)


# Define color gradient for penetration resistance
penetration_resistance_colors = plt.cm.viridis

# Define color gradient for friction ratio
friction_ratio_colors = plt.cm.inferno

# Simplified function to combine continuous segments with the same SBT or similar properties
def combine_segments(df, plot_type, property_name=None, n_bins=10):
    combined_data = []
    for borehole_id in df['borehole_id'].unique():
        sounding_data = df[df['borehole_id'] == borehole_id].sort_values('depth').reset_index(drop=True)
        
        if property_name:
            min_value = df[property_name].min()
            max_value = df[property_name].max()
            bins = np.linspace(min_value, max_value, n_bins + 1)
            categories = np.digitize(sounding_data[property_name], bins) - 1
            categories[categories == n_bins] = n_bins - 1  # Fix any out-of-range indices
        else:
            categories = sounding_data['SBT_index']
        
        start_idx = 0
        for i in range(1, len(sounding_data)):
            if categories[i] != categories[start_idx]:
                combined_data.append({
                    'borehole_id': borehole_id,
                    'start_elevation': sounding_data['elevation'][start_idx],
                    'end_elevation': sounding_data['elevation'][i],
                    plot_type: categories[start_idx]
                })
                start_idx = i
        
        # Add the last segment
        combined_data.append({
            'borehole_id': borehole_id,
            'start_elevation': sounding_data['elevation'][start_idx],
            'end_elevation': sounding_data['elevation'][len(sounding_data)-1],
            plot_type: categories[start_idx]
        })
    
    combined_df = pd.DataFrame(combined_data)
    return combined_df, bins if property_name else None

plot_type = 'friction_ratio'
plot_type = 'penetration_resistance'
# plot_type = 'SBT'

if plot_type == 'SBT':
    lithology_df, _ = combine_segments(lithology_df, plot_type, property_name=None)
    lithology_colors = SBT_colors
else:
    lithology_df, bins = combine_segments(lithology_df, plot_type, property_name=plot_type, n_bins=10)
    lithology_colors = penetration_resistance_colors if plot_type == 'penetration_resistance' else friction_ratio_colors
    lithology_colors = lithology_colors(np.linspace(0, 1, 10))

# Function to create cylinder data
def create_cylinder_data(center_x, center_y, start_elev, end_elev, radius=0.5, num_sides=20):
    theta = np.linspace(0, 2*np.pi, num_sides)
    z = np.array([start_elev, end_elev])
    theta, z = np.meshgrid(theta, z)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y, z

# Define the line for the projected profile view
point1 = [1000, 2039]
point2 = [1100, 2040]

# Function to project boreholes orthogonally onto the vertical plane
def project_onto_plane(x, y, point1, point2):
    if point1 == point2:  # Handle the case where the points are identical
        return x, y
    direction_vector = np.array([point2[0] - point1[0], point2[1] - point1[1] + 1e-16]) # Add a very small amount in the case y's are equal to avoid division by zero
    unit_direction_vector = direction_vector / np.linalg.norm(direction_vector)
    projection = np.dot(np.array([x, y]) - np.array(point1), unit_direction_vector) * unit_direction_vector
    projected_x = point1[0] + projection[0]
    projected_y = point1[1] + projection[1]
    return projected_x, projected_y

# Create 3D plot
fig = plt.figure(figsize=(20, 10))

# Plot view 1 (Isometric view)
ax1 = fig.add_subplot(121, projection='3d')

for bh in collar_df['borehole_id']:
    collar = collar_df[collar_df['borehole_id'] == bh]
    lithologies = lithology_df[lithology_df['borehole_id'] == bh]
    
    for i in range(1, len(lithologies)):
        row = lithologies.iloc[i]
        prev_row = lithologies.iloc[i-1]
        
        start_elevation = prev_row['end_elevation']
        end_elevation = row['end_elevation']
        lithology = row[plot_type]
        
        x = collar['X'].values[0]
        y = collar['Y'].values[0]
        
        # Create cylinder for each lithology interval
        X, Y, Z = create_cylinder_data(x, y, start_elevation, end_elevation, radius=2.5)
        
        if plot_type == 'SBT':
            color = SBT_colors(lithology - 1)  # Adjust for zero-based index
        else:
            color = lithology_colors[lithology]
        
        ax1.plot_surface(X, Y, Z, color=color, alpha=0.75)
        
    # Plot a circle at the top of the borehole
    ax1.scatter([x], [y], [max(lithologies.start_elevation)], color='k', marker='o', s=50)
    
    # Add labels at the top of the borings
    ax1.text(x, y, collar['Z'].values[0], bh, color='black')

# Set labels and plot limits for view 1
ax1.set_xlabel('X (Longitude)')
ax1.set_ylabel('Y (Latitude)')
ax1.set_zlabel('Elevation (ft, NGVD29)')
ax1.set_xlim(collar_df['X'].min() - 10, collar_df['X'].max() + 10)
ax1.set_ylim(collar_df['Y'].min() - 10, collar_df['Y'].max() + 10)
ax1.set_zlim(collar_df['Z'].min() - 110, collar_df['Z'].max() + 10)
ax1.view_init(elev=30, azim=60)

# Draw the vertical projection plane over the specified line
plane_height = collar_df['Z'].max() - collar_df['Z'].min() + 120
x_plane = np.array([point1[0], point2[0]])
y_plane = np.array([point1[1], point2[1]])
z_plane = np.linspace(collar_df['Z'].min() - 110, collar_df['Z'].max() + 10, 10)
X_plane, Z_plane = np.meshgrid(x_plane, z_plane)
Y_plane = np.tile(y_plane, (len(z_plane), 1))

ax1.plot_surface(X_plane, Y_plane, Z_plane, color='lightblue', alpha=0.3)

# Draw the projection line at the bottom of the plot
x_line = np.linspace(point1[0], point2[0], 100)
y_line = np.linspace(point1[1], point2[1], 100)
z_line = np.full_like(x_line, collar_df['Z'].min() - 110)
ax1.plot(x_line, y_line, z_line, c='red', linewidth=2)

# Plot open circles on the projection line in 3D plot
ax1.scatter([point1[0]], [point1[1]], [collar_df['Z'].min() - 110], 
            facecolors='none', edgecolors='r', s=100, marker='o', linewidths=2)
ax1.scatter([point2[0]], [point2[1]], [collar_df['Z'].min() - 110], 
            facecolors='none', edgecolors='r', s=100, marker='^', linewidths=2)

# 2D Projected profile view
ax2 = fig.add_subplot(122)

for bh in collar_df['borehole_id']:
    collar = collar_df[collar_df['borehole_id'] == bh]
    lithologies = lithology_df[lithology_df['borehole_id'] == bh]
    
    x_proj, y_proj = project_onto_plane(collar['X'].values[0], collar['Y'].values[0], point1, point2)
    
    for i in range(1, len(lithologies)):
        row = lithologies.iloc[i]
        prev_row = lithologies.iloc[i-1]
        
        start_elevation = prev_row['end_elevation']
        end_elevation = row['end_elevation']
        lithology = row[plot_type]
        
        bhwidth = 0.01
        
        if plot_type == 'SBT':
            color = SBT_colors(lithology - 1)  # Adjust for zero-based index
        else:
            color = lithology_colors[lithology]
        
        # ax2.fill_between([y_proj-bhwidth, y_proj+bhwidth],
        #                  start_elevation, end_elevation, 
        #                  color=color, edgecolor='black')
        
        # or plot as line segments
        ax2.plot([y_proj, y_proj], [start_elevation, end_elevation], color=color, linewidth=10)
    
    
    # Add labels at the top of the borings
    bhtop = lithologies.start_elevation.max()
    voffset = 0.5
    ax2.text(y_proj, bhtop + voffset, bh, color='black', verticalalignment='bottom')

ax2.set_ylim(ax2.get_ylim())
ax2.set_xlim(ax2.get_xlim())
ax2.plot(ax2.get_xlim(), [ax2.get_ylim()[0]]*2, 'r', lw=3, clip_on=False)
ax2.plot(ax2.get_xlim()[0], ax2.get_ylim()[0], 'ro', mfc='w', mew=3, ms=15, clip_on=False)
ax2.plot(ax2.get_xlim()[1], ax2.get_ylim()[0], 'r^', mfc='w', mew=3, ms=15, clip_on=False)

# Set labels and plot limits for view 2
ax2.set_xlabel('Projected Distance')
ax2.set_ylabel('Elevation (ft)')
ax2.grid(which='both', alpha=0.4)

# Add legend
if plot_type == 'SBT':
    handles = [plt.Line2D([0], [0], color=SBT_colors(i), lw=10) for i in range(9)]
    labels = [f'SBT {i+1}' for i in range(9)]
elif plot_type == 'penetration_resistance':
    handles = [plt.Line2D([0], [0], color=penetration_resistance_colors(0.1), lw=10),
               plt.Line2D([0], [0], color=penetration_resistance_colors(0.9), lw=10)]
    labels = ['Low', 'High']
else:
    handles = [plt.Line2D([0], [0], color=friction_ratio_colors(0.1), lw=10),
               plt.Line2D([0], [0], color=friction_ratio_colors(0.9), lw=10)]
    labels = ['Low', 'High']
    

ax1.legend(handles, labels, loc='best', title="Soil Behavior Type" if plot_type == 'SBT' else plot_type.replace('_', ' ').title())

# # Add colorbar
# if plot_type != 'SBT':
#     mappable = plt.cm.ScalarMappable(cmap=penetration_resistance_colors if plot_type == 'penetration_resistance' else friction_ratio_colors)
#     mappable.set_array(lithology_df[plot_type])
#     cbar = fig.colorbar(mappable, ax=ax1, orientation='vertical', fraction=0.02, pad=0.04)
#     cbar.set_label(plot_type.replace('_', ' ').title())



plt.tight_layout()

fig.savefig("demo2.svg", dpi=300, bbox_inches="tight")

plt.show()
