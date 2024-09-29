import matplotlib.pyplot as plt

def plot_overlapping_boxes(image, depth_map, boxes, detected_classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()

    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        color = "red"
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=3))
        class_name = detected_classes[i]
        ax.text(xmin, ymin, f'{class_name}', fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()
