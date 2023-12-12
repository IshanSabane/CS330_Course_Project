import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    union_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection_area

    iou = intersection_area / union_area
    return iou, (max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4))

# Box 1 coordinates
box1 = [2, 3, 7, 6]

# Box 2 coordinates
box2 = [5, 4, 10, 8]

# Calculate IoU and intersection coordinates
iou, intersection_coords = calculate_iou(box1, box2)

# Plotting
fig, ax = plt.subplots()

# Add rectangles to the plot
rect1 = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1], linewidth=1, edgecolor='r', facecolor='none', label='Box 1')
rect2 = patches.Rectangle((box2[0], box2[1]), box2[2] - box2[0], box2[3] - box2[1], linewidth=1, edgecolor='b', facecolor='none', label='Box 2')

# Add shaded region for intersection
intersection_rect = patches.Rectangle((intersection_coords[0], intersection_coords[1]), intersection_coords[2] - intersection_coords[0], intersection_coords[3] - intersection_coords[1], linewidth=0, edgecolor='none', facecolor='gray', alpha=0.5, label='Intersection')

ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(intersection_rect)

# Set axis limits
plt.xlim(0, 15)
plt.ylim(0, 10)

# Add legend
plt.legend()

# Display IoU
plt.title(f'IoU: {iou:.2f}')

# Show the plot
plt.show()
