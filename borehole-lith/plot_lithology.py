import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
collar_df = pd.read_csv('collar.csv')
lithology_df = pd.read_csv('lithology.csv')

# Define colors for lithologies
lithology_colors = {
    'Sand': 'yellow',
    'Clay': 'brown',
    'Gravel': 'gray',
    'Silt': 'green',
    'Limestone': 'blue'
}

# Function to create cylinder data
def create_cylinder_data(center_x, center_y, start_elev, end_elev, radius=0.5, num_sides=20):
    theta = np.linspace(0, 2*np.pi, num_sides)
    z = np.array([start_elev, end_elev])
    theta, z = np.meshgrid(theta, z)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y, z

# Define the line for the projected profile view

point1 = (1100, 2000)
point2 = (1100, 2100)

point1 = (1000, 2000)
point2 = (1100, 2100)

point1 = (1000, 2099)
point2 = (1100, 2100)

point1 = [1000, 2039]
point2 = [1100, 2040]


# Function to project boreholes orthogonally onto the vertical plane
def project_onto_plane(x, y, point1, point2):
    if point1 == point2:  # Handle the case where the points are identical
        return x, y
    direction_vector = np.array([point2[0] - point1[0], 
                                 point2[1] - point1[1] + 1e-16]) # Add a very small amount in the case y's are equal to avoid division by zero
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
    
    
    for _, row in lithologies.iterrows():
        start_elevation = row['start_elevation']
        end_elevation = row['end_elevation']
        lithology = row['lithology']
        
        x = collar['X'].values[0]
        y = collar['Y'].values[0]
        
        # Create cylinder for each lithology interval
        X, Y, Z = create_cylinder_data(x, y, start_elevation, end_elevation, radius=2.5)
        ax1.plot_surface(X, Y, Z, color=lithology_colors[lithology], alpha=0.75)
        
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
# ax1.scatter([point1[0], point2[0]], [point1[1], point2[1]], [collar_df['Z'].min() - 110, collar_df['Z'].min() - 110], 
#             facecolors='none', edgecolors='r', s=100, marker='o', linewidths=2)
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
    
    for _, row in lithologies.iterrows():
        start_elevation = row['start_elevation']
        end_elevation = row['end_elevation']
        lithology = row['lithology']
        
        bhwidth = 0.01
        ax2.fill_between([y_proj-bhwidth, y_proj+bhwidth],
                         start_elevation, end_elevation, 
                         color=lithology_colors[lithology], edgecolor='black')
    
    
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
handles = [plt.Line2D([0], [0], color=color, lw=10) for color in lithology_colors.values()]
labels = lithology_colors.keys()
# fig.legend(handles, labels, loc='upper right', title="Lithologies")
ax1.legend(handles, labels, loc='best', title="Lithologies")

plt.tight_layout()

fig.savefig("demo.svg", dpi=300, bbox_inches="tight")

plt.show()
