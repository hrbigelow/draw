## Tensorflow v2.6 implementation of the Deep Recurrent Attentive Writer 

This is a TF v2.6 implementation of the Deep Recurrent Attentive Writer from
https://arxiv.org/pdf/1502.04623.pdf.  It is based on Eric Jang's
implementation [here](https://github.com/ericjang/draw).

## Usage


`python draw.py --help`

`python draw.py --data_dir=/path/to/data/tmp --learning_rate=5e-4 --start_step=9000 --ckpt_template=/path/to/ckpt/run2.%.ckpt --tboard_logdir=/path/to/tb/log/run2/`

To generate new data using a trained model:

`python gen_data.py --ckpt_path=ckpt/draw.17500.ckpt --img_path=/path/to/results/draw.17500`

This will create a series of images "imagined" by the trained decoder model,
one at each timestep during the iterative generation process.  You can then
make them into a gif with:

`convert -delay 10 -loop 0 draw.17500_?.png draw.17500_??.png draw.17500.gif`
`animate draw.17500.gif`









