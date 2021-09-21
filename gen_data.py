import numpy as np
import tensorflow as tf
from draw import Draw
from plot_data import draw_steps
from hparams import setup_hparams
import sys
import fire

def main(hps, ckpt_path, img_path, **kwargs):
    hps = setup_hparams(hps, kwargs)
    model = Draw(hps)
    model.load_weights(ckpt_path)

    # print(model)
    model.record(True)
    x = model.generate()
    canvases = np.array(model.canvases)

    draw_steps(img_path, canvases)

if __name__ == '__main__':
    fire.Fire(main)

