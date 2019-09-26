"""largely adapted from:
    Matplotlib Animation Example
    author: Jake Vanderplas
    email: vanderplas@astro.washington.edu
    website: http://jakevdp.github.com
    license: BSD
"""
import os
import torch
import numpy
import cv2

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import animation

def make_animations(dir, nframes, use_color=False):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes()
    x0 = torch.load(os.path.join(dir, 'depth')+str(0).zfill(6)+'.pt').numpy()
    img = ax.imshow(x0, cmap='gray')

    # animation function.  This is called sequentially
    def animate(i):
        x = torch.load(os.path.join(dir, 'depth')+str(i).zfill(6)+'.pt').numpy()
        img.set_data(x)
        return img,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, frames=nframes, interval=20, blit=True)
    anim.save(os.path.join(dir,'animation.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])

    if use_color:

        # do the same again, but with color
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes()
        # x0 = torch.load(os.path.join(dir, 'depth')+str(0).zfill(6)+'.pt').numpy()
        x0 = cv2.imread(os.path.join(dir,'img')+str(0).zfill(6)+'.png')
        img = ax.imshow(x0)

        # animation function.  This is called sequentially
        def animate(i):
            x = cv2.imread(os.path.join(dir,'img')+str(i).zfill(6)+'.png')
            img.set_data(x)
            return img,

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, frames=nframes, interval=20, blit=True)
        anim.save(os.path.join(dir,'color_animation.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])
        # plt.show()

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='directory to animate')
    parser.add_argument('--n', type=int, help='how many to visualize')
    parser.add_argument('--color', action='store_true', help='show color?')
    args=parser.parse_args()
    make_animations(args.dir, args.n, use_color=args.color)
