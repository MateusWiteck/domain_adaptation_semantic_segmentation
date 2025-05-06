import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from GTA5Label import GTA5Labels_TaskCV2017  # adjust the import

GTA5_LABEL_NAMES = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'light',
    7: 'sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motocycle',
    18: 'bicycle',
    255: 'ignore',  
}


def visualize_label_with_colors(label_array):
    """
    Visualize a class ID label map using the GTA5 color palette with a legend.

    Args:
        label_array (np.ndarray): 2D array with class IDs.
    """
    # Build the colormap (num_classes x 3)
    num_classes = len(GTA5Labels_TaskCV2017.list_)
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    for label in GTA5Labels_TaskCV2017.list_:
        colormap[label.ID] = label.color

    # Initialize a color image (H x W x 3)
    color_label = np.zeros((label_array.shape[0], label_array.shape[1], 3), dtype=np.uint8)

    # Fill in the colors
    for class_id in np.unique(label_array):
        if class_id == 255:
            continue  # Skip ignore label
        mask = label_array == class_id
        color_label[mask] = colormap[class_id]

    # Plot the image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(color_label)
    ax.set_title('Label (Color)')
    ax.axis('off')

    # Prepare legend handles (only include labels present in the image)
    unique_ids = [cls_id for cls_id in np.unique(label_array) if cls_id != 255]
    legend_elements = []
    for cls_id in unique_ids:
        label = next(l for l in GTA5Labels_TaskCV2017.list_ if l.ID == cls_id)
        label_name = GTA5_LABEL_NAMES.get(cls_id, f'Class {cls_id}')
        legend_elements.append(
            Patch(
                facecolor=np.array(label.color) / 255,
                edgecolor='black',
                label=f'{cls_id}: {label_name}'
            )
        )

    # Place the legend outside the image
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.show()
