#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This module contains vizualisation tools
# Ayoub Belhadji
# 05/12/2025

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_trajectories_box(
        objective_function, trajectories,
        frame_step, bounds, save_path=None,
        writer='imagemagick'
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
    ax.set_ylim(bounds[0] - 1, bounds[1] + 1)

    # Create a background grid for the objective function
    x = np.linspace(bounds[0] - 1, bounds[1] + 1, 100)
    y = np.linspace(bounds[0] - 1, bounds[1] + 1, 100)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = torch.exp(objective_function(torch.tensor(XY))
                  ).detach().numpy().reshape(100, 100)

    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    fig.colorbar(contour, ax=ax)  # Add colorbar to show the color map key
    scat = ax.scatter([], [], s=50, color='red')
    # iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='white')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        # iteration_text.set_text('')
        ax.set_title('Iteration = 0')
        return scat,

    def update(frame):
        scat.set_offsets(trajectories[frame])
        # iteration_text.set_text(f'Iteration = {frame}')
        ax.set_title(f'Iteration = {frame}')
        return scat,

    frames_list = range(0, len(trajectories), frame_step)

    ani = animation.FuncAnimation(
        fig, update, frames=frames_list, init_func=init, blit=True, interval=100
    )

    if save_path:
        ani.save(save_path, writer=writer)
    else:
        plt.show()

    plt.close()


def animate_trajectories_box_(objective_function, trajectories, bounds, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
    ax.set_ylim(bounds[0] - 1, bounds[1] + 1)

    # Create a background grid for the objective function
    x = np.linspace(bounds[0] - 1, bounds[1] + 1, 100)
    y = np.linspace(bounds[0] - 1, bounds[1] + 1, 100)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = objective_function(torch.tensor(XY)).detach().numpy().reshape(100, 100)

    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    fig.colorbar(contour, ax=ax)  # Add colorbar to show the color map key
    scat = ax.scatter([], [], s=50, color='red')
    # iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='white')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        # iteration_text.set_text('')
        ax.set_title('Iteration = 0')
        return scat,

    def update(frame):
        scat.set_offsets(trajectories[frame])
        # iteration_text.set_text(f'Iteration = {frame}')
        ax.set_title(f'Iteration = {frame}')
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(trajectories), init_func=init, blit=True, interval=100
    )

    if save_path:
        ani.save(save_path, writer='imagemagick')
    else:
        plt.show()

    plt.close()


def animate_trajectories_other(objective_function, trajectories, bounds, save_path=None):
    # To be finished: animations in other shapes?! or other formats?
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0] - 1, bounds[1] + 1)
    ax.set_ylim(bounds[0] - 1, bounds[1] + 1)

    # Create a background grid for the objective function
    x = np.linspace(bounds[0] - 1, bounds[1] + 1, 100)
    y = np.linspace(bounds[0] - 1, bounds[1] + 1, 100)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = objective_function(torch.tensor(XY)).detach().numpy().reshape(100, 100)

    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    fig.colorbar(contour, ax=ax)  # Add colorbar to show the color map key
    scat = ax.scatter([], [], s=50, color='red')
    # iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='white')

    def init():
        scat.set_offsets(np.empty((0, 2)))
        # iteration_text.set_text('')
        ax.set_title('Iteration = 0')
        return scat,

    def update(frame):
        scat.set_offsets(trajectories[frame])
        # iteration_text.set_text(f'Iteration = {frame}')
        ax.set_title(f'Iteration = {frame}')
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(trajectories), init_func=init, blit=True, interval=100
    )

    if save_path:
        ani.save(save_path, writer='imagemagick')
    else:
        plt.show()

    plt.close()
