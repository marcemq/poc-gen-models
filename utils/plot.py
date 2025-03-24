import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.colors import ListedColormap
from utils.utils import create_directory

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

def plot_checkerboard_over_time(xt_over_time, cboard):
    title =  f"Sampling of xt over time"
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
    gif_name = "images/xt_over_time.gif"
    ani.save(gif_name, writer=PillowWriter(fps=5))
    plt.close(fig)

    logging.info("Xt over time gif saved at images dir")