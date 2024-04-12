#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
import os
from pathlib import Path

class Cell:
    def __init__(self, cell_id, position):
        self.cell_id = cell_id
        self.position = position
        self.history = [position]  # Track position history

    def update_position(self, new_position):
        self.position = new_position
        self.history.append(new_position)

class CellTracker:
    def __init__(self):
        self.cells = []  # List to store tracked cells

    def track_cells(self, frames):
        for frame_number, frame in enumerate(frames):
            for cell_id, cell_position in frame.items():
                if not self.cells:  # If no cells are tracked yet, create a new cell
                    new_cell = Cell(cell_id, cell_position)
                    self.cells.append(new_cell)
                else:
                    # Find the closest cell to the current detected cell
                    closest_cell = min(self.cells, key=lambda cell: self.distance(cell.position, cell_position))

                    # If the distance between the closest cell and the detected cell is less than a threshold,
                    # consider it as the same cell and update its position
                    threshold = 20  # Adjust this value based on your requirements
                    if self.distance(closest_cell.position, cell_position) < threshold:
                        closest_cell.update_position(cell_position)
                    else:
                        # Otherwise, create a new cell
                        new_cell = Cell(cell_id, cell_position)
                        self.cells.append(new_cell)

    @staticmethod
    def distance(pos1, pos2):
        # Calculate Euclidean distance between two positions
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

# Set the path to the folder containing the segmented frames
segmented_frames_dir = '/Users/daniellegermann/Desktop/SegmentedFrames'

all_frames = []

# Loop through the segmented frames and process them
for frame_file in sorted(os.listdir(segmented_frames_dir)):
    frame_path = os.path.join(segmented_frames_dir, frame_file)
    segmented_image = cv2.imread(frame_path, 0)

    # Blob detection
    contours, hierarchy = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a dictionary to store cell positions for the current frame
    current_frame = {}

    # Loop through the contours and extract features
    # Loop through the contours and extract features
    for contour in contours:
        # Calculate the moments
        M = cv2.moments(contour)

        # Calculate the centroid
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            # Skip the contour if the area is 0
            continue

        # Store the cell position in the current frame dictionary
        current_frame[len(current_frame)] = (cx, cy)

    # Add the current frame to the list of all frames
    all_frames.append(current_frame)

# Create an instance of the CellTracker class and track the cells
cell_tracker = CellTracker()
cell_tracker.track_cells(all_frames)

# Access the tracked cells and their position histories
for cell in cell_tracker.cells:
    print(f"Cell ID: {cell.cell_id}, Position History: {cell.history}")


# In[10]:


import matplotlib.pyplot as plt

# Assuming 'cell_tracker' is the instance of CellTracker class containing tracked cells
first_cell = cell_tracker.cells[0]  # Pick the first tracked cell
trajectory = first_cell.history  # Get the trajectory of the first cell

# Extract x and y coordinates from the trajectory
x_coords = [pos[0] for pos in trajectory]
y_coords = [pos[1] for pos in trajectory]

# Calculate plot limits
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)
margin = 100  # Add some margin to the plot limits

# Plot the trajectory using matplotlib
plt.plot(x_coords, y_coords, marker='o', linestyle='-')
plt.xlim(x_min - margin, x_max + margin)  # Set x-axis limits with margin
plt.ylim(y_min - margin, y_max + margin)  # Set y-axis limits with margin
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectory of the First Tracked Cell')
plt.grid(True)
plt.show()


# In[ ]:




