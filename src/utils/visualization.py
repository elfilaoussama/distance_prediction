import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_depth_with_boxes(depth_map, depth_data):
    """
    Plots the depth map with bounding boxes overlayed.
    
    Args:
        depth_map (numpy.ndarray): The depth map to visualize.
        depth_data (pandas.DataFrame): DataFrame containing bounding box coordinates, depth statistics, and class labels.
    """
    # Normalize the depth map for better visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 6))

    # Display the depth map
    ax.imshow(depth_map_normalized, cmap='plasma')  # You can change the colormap as desired
    ax.axis('off')  # Hide the axes

    # Loop through the DataFrame and add rectangles
    for index, row in depth_data.iterrows():
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
        class_label = row['class']
        score = row['depth_mean']  # or whichever statistic you prefer to display

        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='yellow', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add a text label
        ax.text(xmin, ymin - 5, f'{class_label}: {score:.2f}', color='white', fontsize=12, weight='bold')

    plt.title('Depth Map with Object Detection Bounding Boxes', fontsize=16)
    plt.show()

