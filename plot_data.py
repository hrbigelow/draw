# takes data saved by DRAW model and generates animations
# example usage: python plot_data.py noattn /tmp/draw/draw_data.npy

import matplotlib
import fire
import sys
import numpy as np

interactive=False # set to False if you want to write images to file

if not interactive:
    matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt


def xrecons_grid(X):
    """
    plots canvas for single time step

    assumes features = wxA images
    batch is assumed to be a square number
    """
    # num rows/cols, height, width, num channels
    nrow, ncol, h, w, nch = tuple(X.shape)
    padsize, padval = 1, 0.5
    pw=w+2*padsize
    ph=h+2*padsize
    img=np.ones((nrow * ph, ncol * pw, nch))*padval
    for r in range(nrow):
        begr=r*ph+padsize
        endr=begr+h
        for c in range(ncol):
            begc=c*pw+padsize
            endc=begc+w
            img[begr:endr,begc:endc]=X[r,c,...]
    return img

def draw_steps(prefix, nrow, ncol, canvases):
    T, batch_size, num_chan, img_size = canvases.shape
    h = w = int(np.sqrt(img_size))
    X=1.0/(1.0+np.exp(-canvases)) # x_recons=sigmoid(canvas)
    X = np.transpose(X, (0,1,3,2))
    X=X.reshape((T,nrow,ncol,h,w,num_chan))
    if interactive:
        f,arr=plt.subplots(1,T)
    for t in range(T):
        img=xrecons_grid(X[t,...])
        if interactive:
            if num_chan == 1:
                arr[t].matshow(img,cmap=plt.cm.gray)
            else:
                arr[t].imshow(img)
            arr[t].set_xticks([])
            arr[t].set_yticks([])
        else:
            if num_chan == 1:
                plt.matshow(img,cmap=plt.cm.gray)
            else:
                plt.imshow(img)
            imgname='%s_%d.png' % (prefix,t) # you can merge using imagemagick, i.e. convert -delay 10 -loop 0 *.png mnist.gif
            plt.savefig(imgname)
            print(imgname)


