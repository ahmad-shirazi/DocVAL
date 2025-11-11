"""
Visualization utilities for DocVAL predictions
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Dict
import numpy as np

def visualize_prediction(
    image: Image.Image,
    prediction: Dict,
    ground_truth: Dict = None,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize prediction with bounding box overlay
    
    Args:
        image: Document image
        prediction: Prediction dict with 'answer' and 'bbox'
        ground_truth: Optional ground truth for comparison
        save_path: Optional path to save visualization
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display image
    ax.imshow(image)
    
    # Draw predicted bbox (red)
    pred_bbox = prediction['bbox']
    x1, y1, x2, y2 = pred_bbox
    width = x2 - x1
    height = y2 - y1
    
    rect = patches.Rectangle(
        (x1, y1), width, height,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        label=f"Prediction: {prediction['answer']}"
    )
    ax.add_patch(rect)
    
    # Draw ground truth bbox (green) if provided
    if ground_truth:
        gt_bbox = ground_truth['bbox']
        x1, y1, x2, y2 = gt_bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor='green',
            facecolor='none',
            label=f"Ground Truth: {ground_truth['answer']}"
        )
        ax.add_patch(rect)
    
    ax.axis('off')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_cot_steps(
    cot_steps: List[str],
    save_path: str = None
):
    """
    Visualize chain-of-thought reasoning steps
    
    Args:
        cot_steps: List of reasoning steps
        save_path: Optional path to save visualization
    """
    fig, ax = plt.subplots(figsize=(10, len(cot_steps) * 1.5))
    
    ax.axis('off')
    
    # Display steps
    y_pos = 0.9
    for i, step in enumerate(cot_steps, 1):
        ax.text(
            0.05, y_pos,
            f"Step {i}: {step}",
            fontsize=12,
            verticalalignment='top',
            wrap=True
        )
        y_pos -= 0.15
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()

