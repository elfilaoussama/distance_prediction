"""
Created on Wed Jun  1 06:08:23 2022
@author: ODD team

Edited by our team : Sat Oct 5 11:27 2024

# Process function - Key Functionalities
This script processes object detection outputs (bounding boxes and depth maps) 
to compute depth statistics and handle overlapping bounding boxes.
"""

import pandas as pd
import numpy as np
from scipy import stats


class PROCESSING :
    def __init__(self):
        pass

    def process_detections(self, scores, boxes, depth_map, detr):
        """
        Processes object detections, computes depth statistics, and handles overlapping bounding boxes.
        
        Args:
            scores (list): List of class prediction scores from the object detection model.
            boxes (numpy.ndarray): Bounding boxes in the format [xmin, ymin, xmax, ymax].
            depth_map (numpy.ndarray): Depth map corresponding to the image.
            detr (object): Pretrained object detection model (e.g., detr) containing class information.
        
        Returns:
            pandas.DataFrame: Processed dataset containing bounding box coordinates, 
                            depth statistics, and object class.
        """
        # Initialize a DataFrame for storing results
        self.data = pd.DataFrame(columns=['xmin','ymin','xmax','ymax','width', 'height','depth_mean_trim','depth_mean','depth_median', 'class', 'rgb'])

        # Iterate over detected bounding boxes and their corresponding scores
        for p, (xmin, ymin, xmax, ymax) in zip(scores, boxes.tolist()):
            # Identify the class with the highest score
            detected_class = p.argmax()
            class_label = detr.CLASSES[detected_class]

            # Filter for relevant object classes
            if class_label == 'motorcycle':
                class_label = 'bicycle'
            elif class_label == 'bus':
                class_label = 'train'
            elif class_label not in ['person', 'truck', 'car', 'bicycle', 'train']:
                class_label = 'Misc'

            if class_label in ['Misc', 'person', 'truck', 'car', 'bicycle', 'train']:
                # Assign RGB color for the detected class
                class_index = ['Misc', 'person', 'truck', 'car', 'bicycle', 'train'].index(class_label)
                r, g, b = detr.COLORS[class_index]
                rgb = (r * 255, g * 255, b * 255)

                # Calculate bounding box dimensions
                width, height = xmax - xmin, ymax - ymin
                xmin, ymin = max(0, int(xmin)), max(0, int(ymin))

                # Compute depth statistics within the bounding box
                bbox_depth = depth_map[int(ymin):int(ymax), int(xmin):int(xmax)]
                depth_mean = bbox_depth.mean()
                depth_median = np.median(bbox_depth)
                depth_trimmed_mean = stats.trim_mean(bbox_depth.flatten(), 0.2)
                #depth_max = bbox_depth.max()

                # Store the calculated data in the DataFrame
                new_row = pd.DataFrame([[xmin, ymin, xmax, ymax, width, height, depth_trimmed_mean ,depth_mean, depth_median, class_label, rgb]], 
                                    columns=self.data.columns)
                self.data = pd.concat([self.data, new_row], ignore_index=True)

        # Handle overlapping bounding boxes
        self.handle_overlaps(depth_map)

        return self.data

        
    def handle_overlaps(self, depth_map):
        """
        Handles overlapping bounding boxes by removing the farther object 
        or recalculating depth statistics for the overlapping region.

        Args:
            depth_map (numpy.ndarray): Depth map corresponding to the image.
        """
        # Reset the index for easy iteration
        self.data.reset_index(drop=True, inplace=True)

        # Lists to track the bounding box coordinates
        xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []

        # Loop through each bounding box in the dataset
        for index, (xmin, ymin, xmax, ymax) in self.data[['xmin', 'ymin', 'xmax', 'ymax']].iterrows():
            xmin_list.insert(0, xmin)
            ymin_list.insert(0, ymin)
            xmax_list.insert(0, xmax)
            ymax_list.insert(0, ymax)

            # Compare the current bounding box with all previous ones
            for i in range(len(xmin_list) - 1):
                # Check Y-axis overlap
                y_range1 = np.arange(int(ymin_list[0]), int(ymax_list[0]) + 1)
                y_range2 = np.arange(int(ymin_list[i + 1]), int(ymax_list[i + 1]) + 1)
                y_intersection = np.intersect1d(y_range1, y_range2)

                if len(y_intersection) >= 1:
                    # Check X-axis overlap
                    x_range1 = np.arange(int(xmin_list[0]), int(xmax_list[0]) + 1)
                    x_range2 = np.arange(int(xmin_list[i + 1]), int(xmax_list[i + 1]) + 1)
                    x_intersection = np.intersect1d(x_range1, x_range2)

                    if len(x_intersection) >= 1:
                        # Calculate the areas of the bounding boxes and their intersection
                        area1 = (y_range1.max() - y_range1.min()) * (x_range1.max() - x_range1.min())
                        area2 = (y_range2.max() - y_range2.min()) * (x_range2.max() - x_range2.min())
                        area_intersection = (y_intersection.max() - y_intersection.min()) * (x_intersection.max() - x_intersection.min())

                        # If more than 70% overlap, remove the farther object
                        if area_intersection / area1 >= 0.70 or area_intersection / area2 >= 0.70:
                            if area1 < area2:
                                self.data.drop(index=index, inplace=True)
                            else:
                                self.data.drop(index=index - (i + 1), inplace=True)

                        # If partial overlap, recalculate depth for the overlapping region
                        elif area_intersection / area1 > 0 or area_intersection / area2 > 0:
                            # Convert to integers for indexing
                            y_min_idx = int(y_intersection.min())
                            y_max_idx = int(y_intersection.max())
                            x_min_idx = int(x_intersection.min())
                            x_max_idx = int(x_intersection.max())

                            if area1 < area2:
                                # Check bounds before slicing
                                if (0 <= y_min_idx < depth_map.shape[0]) and (0 <= y_max_idx < depth_map.shape[0]) and \
                                (0 <= x_min_idx < depth_map.shape[1]) and (0 <= x_max_idx < depth_map.shape[1]):
                                    depth_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = np.nan
                                    bbox_depth = depth_map[int(ymin_list[0]):int(ymax_list[0]), int(xmin_list[0]):int(xmax_list[0])]
                                    self.data.at[index, 'depth_mean'] = np.nanmean(bbox_depth)
                                else:
                                    print("Index out of bounds for depth map:", y_min_idx, y_max_idx, x_min_idx, x_max_idx)
                            else:
                                # Similar bounds checking for the other box
                                if (0 <= y_min_idx < depth_map.shape[0]) and (0 <= y_max_idx < depth_map.shape[0]) and \
                                (0 <= x_min_idx < depth_map.shape[1]) and (0 <= x_max_idx < depth_map.shape[1]):
                                    depth_map[y_min_idx:y_max_idx, x_min_idx:x_max_idx] = np.nan
                                    bbox_depth = depth_map[int(ymin_list[i + 1]):int(ymax_list[i + 1]), int(xmin_list[i + 1]):int(xmax_list[i + 1])]
                                    self.data.at[index - (i + 1), 'depth_mean'] = np.nanmean(bbox_depth)
                                else:
                                    print("Index out of bounds for depth map:", y_min_idx, y_max_idx, x_min_idx, x_max_idx)

        # Reset index after removing rows
        self.data.reset_index(drop=True, inplace=True)
