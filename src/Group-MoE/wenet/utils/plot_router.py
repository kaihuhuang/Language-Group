import matplotlib.pyplot as plt
import torch
import os
from matplotlib.ticker import FixedLocator, FixedFormatter


def plot_tensor_scatter(tensor, save_dir="exp"):
    indices = torch.arange(len(tensor))
    values = tensor.view(-1).tolist()
    label_map = {0: 'Zh', 1: 'En'}
    mapped_values = [label_map[val] for val in values]
    colors = ['r' if val == 0 else 'g' for val in values]
    plt.figure(figsize=(6, 2))
    plt.scatter(indices, values, s=5, c=colors, marker='o')
    plt.title('Language Router Routing Analysis')
    plt.xlabel('Frame Index')
    plt.ylabel('Language Group')
    plt.gca().yaxis.set_major_locator(FixedLocator([0, 1]))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(['Zh', 'En']))

    plt.grid(True)
    plt.tight_layout()
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'router.svg')
        plt.savefig(save_path, format='svg')
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()
