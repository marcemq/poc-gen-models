import logging, os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.colors import ListedColormap
from utils.utils import create_directory
from utils.data import inverse_transform
from torchvision.utils import make_grid

def plot_epoch_loss(epoch_loss, output_dir, model_prefix):
    """Plot and save the training loss curve over epochs."""
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, marker='o', linewidth=2)
    plt.title(f"{model_prefix} Training mean loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Epoch Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save path
    plot_path = os.path.join(output_dir, f"{model_prefix}_epoch_loss_curve.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    logging.info(f"Epoch loss plot saved at: {plot_path}")

def plot_checkerboard(cboard, saveImg=True):
    create_directory("images")
    # Plot the checkerboard pattern
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cboard.checkerboard_pattern, extent=(cboard.x_min, cboard.x_max, cboard.y_min, cboard.y_max), origin="lower", cmap=ListedColormap(["purple", "yellow"]))

    # Plot sampled points
    ax.scatter(cboard.sampled_points[:, 0], cboard.sampled_points[:, 1], color="red", marker="o", s=15)
    ax.set_title("GT Checkerboard", fontsize=15, fontweight='bold')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    if saveImg:
        fig.savefig("images/checkerboard.png", format='png', bbox_inches='tight')

def plot_checkerboard_over_time(xt_over_time, model_prefix, output_dir, plot_name, plot_fps, cboard):
    title =  f"Sampling of checkerboard over time with {model_prefix.upper()} model."
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot checkerboard background
    ax.imshow(cboard.checkerboard_pattern, extent=(cboard.x_min, cboard.x_max, cboard.y_min, cboard.y_max), origin="lower", cmap=ListedColormap(["purple", "yellow"]))
    # Initial scatter plot for sampled points
    ax.scatter(cboard.sampled_points[:, 0], cboard.sampled_points[:, 1], color="red", marker="o", s=15)
    # Scatter plot for dynamic sampling (initially empty)
    scatter_plot = ax.scatter([], [], color="green", marker="o", label="Generated Samples", s=15)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title(title, fontsize=15, fontweight='bold')

    frame_text = ax.text(0.5, -0.125, '', transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    def update(frame):
        t, xt = xt_over_time[frame]
        frame_text.set_text(f't = {t:.2f}')
        scatter_plot.set_offsets(xt.cpu().numpy())

    # Set up animation for the current sequence
    ani = animation.FuncAnimation(fig, update, frames=len(xt_over_time), repeat=True)

    # Saving output
    gif_name = os.path.join(output_dir, plot_name.format(model_prefix))
    ani.save(gif_name, writer=PillowWriter(fps=plot_fps))
    plt.close(fig)

    logging.info(f"Checkerboard over time gif saved at {output_dir} dir")

def plot_butterflies_over_time(xt_over_time, model_prefix, output_dir, plot_name, plot_fps):
    title =  f"Sampling of butterflies over time with {model_prefix.upper()} model."
    fig, ax = plt.subplots(figsize=(7, 7))

    # Apply inverse transform to each (t, xt) tuple
    xt_over_time = [(t, inverse_transform(xt)) for t, xt in xt_over_time]

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    frame_text = ax.text(0.5, -0.125, '', transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

    def update(frame):
        t, xt = xt_over_time[frame]
        frame_text.set_text(f't = {t:.2f}')
        grid_img = make_grid(xt, nrow=4, padding=True, pad_value=1, normalize=True)
        grid_img_np = grid_img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(grid_img_np)

    # Set up animation for the current sequence
    ani = animation.FuncAnimation(fig, update, frames=len(xt_over_time), repeat=True)

    # Saving output gif
    gif_name = os.path.join(output_dir, plot_name.format(model_prefix))
    ani.save(gif_name, writer=PillowWriter(fps=plot_fps))
    plt.close(fig)

    logging.info(f"Butterflies over time gif saved at {output_dir} dir")