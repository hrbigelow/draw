#!/usr/bin/env python

"""
Adapted from Eric Jang's implementation from 2016, upgraded to TF2
"""

import os
import sys
import fire
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

import numpy as np

A,B = 28,28 # image width,height
img_size = B*A # the canvas size
enc_size = 256 # number of hidden units / output size in LSTM
dec_size = 256
read_n = 2 # read glimpse grid width/height
write_n = 5 # write glimpse grid width/height
# read_size = 2*read_n*read_n if FLAGS.read_attn else 2*img_size
z_size=100 # QSampler output size
T=64 # MNIST generation sequence length
batch_size=100 # training minibatch size
train_iters=10000
eps=1e-8 # epsilon for numerical stability

## BUILD MODEL ## 
# x = tf.keras.Input(shape=(z_size,), batch_size=batch_size, dtype=tf.float32) 

class Linear(layers.Dense):
    def __init__(self, output_dim):
        super(Linear, self).__init__(units=output_dim, 
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros')


def filterbank(gx, gy, sigma2, delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
    Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keepdims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keepdims=True),eps)
    return Fx,Fy


class AttnWindow(tf.keras.layers.Layer):
    def __init__(self):
        super(AttnWindow, self).__init__()
        self.calcParams = Linear(5)

    def call(self, h_dec, N):
        params = self.calcParams(h_dec)
        gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params,5,1)
        gx=(A+1)/2*(gx_+1)
        gy=(B+1)/2*(gy_+1)
        sigma2=tf.exp(log_sigma2)
        delta=(max(A,B)-1)/(N-1)*tf.exp(log_delta) # batch x N
        return filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)


## READ ## 
class ReadAttn(layers.Layer):
    def __init__(self, do_attn=True, **kwargs):
        super(ReadAttn, self).__init__(**kwargs)
        self.attn_window = AttnWindow()
        self.do_attn = do_attn

    def call(self, x, x_hat, h_dec_prev):
        if not self.do_attn:
            return tf.concat([x, x_hat], 1)

        def filter_img(img,Fx,Fy,gamma,N):
            Fxt=tf.transpose(Fx,perm=[0,2,1])
            img=tf.reshape(img,[-1,B,A])
            glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
            glimpse=tf.reshape(glimpse,[-1,N*N])
            return glimpse*tf.reshape(gamma,[-1,1])

        N=read_n
        Fx,Fy,gamma = self.attn_window(h_dec_prev, N)
        x=filter_img(x,Fx,Fy,gamma,N) # batch x (read_n*read_n)
        x_hat=filter_img(x_hat,Fx,Fy,gamma,read_n)
        return tf.concat([x,x_hat], 1) # concat along feature axis


## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##
class SampleQ(layers.Layer):
    def __init__(self, **kwargs):
        super(SampleQ, self).__init__(**kwargs)
        self.mu = Linear(z_size)
        self.logsigma = Linear(z_size)

    def call(self, h_enc, noise):
        _mu = self.mu(h_enc)
        _logsigma = self.logsigma(h_enc)
        _sigma = tf.exp(_logsigma)
        return (_mu + _sigma * noise, _mu, _logsigma, _sigma)


## WRITER ## 
class WriteAttn(layers.Layer):
    def __init__(self, do_attn=True, **kwargs):
        super(WriteAttn, self).__init__(**kwargs)
        write_size = write_n*write_n if do_attn else img_size
        self.calcW = Linear(write_size)
        self.attn_window = AttnWindow()
        self.do_attn = do_attn

    def call(self, h_dec):
        if not self.do_attn:
            return self.calcW(h_dec)

        N=write_n
        w = self.calcW(h_dec)
        w=tf.reshape(w,[batch_size,N,N])
        Fx,Fy,gamma=self.attn_window(h_dec, N)
        Fyt=tf.transpose(Fy,perm=[0,2,1])
        wr=tf.matmul(Fyt,tf.matmul(w,Fx))
        wr=tf.reshape(wr,[batch_size,B*A])
        #gamma=tf.tile(gamma,[1,B*A])
        return wr*tf.reshape(1.0/gamma,[-1,1])



class TimeStep(layers.Layer):
    def __init__(self, read_attn=True, write_attn=True, **kwargs):
        super(TimeStep, self).__init__(**kwargs)
        self.read = ReadAttn(read_attn, name='read') 
        self.encode = layers.LSTMCell(enc_size, name='encoder')
        self.sample = SampleQ(name='sample')
        self.decode = layers.LSTMCell(dec_size, name='decoder')
        self.write = WriteAttn(write_attn, name='write')

    def call(self, x, canvas, enc_states, dec_states, h_dec):
        x_hat = x - tf.sigmoid(canvas)
        r = self.read(x, x_hat, h_dec)
        enc_inputs = tf.concat([r, h_dec], 1)
        h_enc, enc_states = self.encode(enc_inputs, enc_states)
        noise = tf.random.normal((batch_size, z_size), mean=0, stddev=1) 
        z, mus, logsigmas, sigmas = self.sample(h_enc, noise)
        h_dec, dec_states = self.decode(z, dec_states)
        written = self.write(h_dec)
        new_canvas = canvas + written

        # loss term
        # print('mu, sigma mean: %f %f' % 
                # (tf.reduce_mean(mus), tf.reduce_mean(sigmas)))

        mu2 = tf.square(mus)
        sigma2 = tf.square(sigmas)

        # https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
        kl_term = 0.5 * tf.reduce_sum(mu2 + sigma2 - 1 - 2 * logsigmas, 1)
        self.add_loss(kl_term)

        return new_canvas, enc_states, dec_states, h_dec

    def generate(self, canvas, z, dec_states):
        h_dec, dec_states = self.decode(z, dec_states)
        written = self.write(h_dec)
        canvas = canvas + written
        return canvas, dec_states



## DRAW MODEL ## 



class Draw(tf.keras.Model):
    def __init__(self, read_attn=True, write_attn=True, **kwargs):
        super(Draw, self).__init__(**kwargs)
        self.step = TimeStep(read_attn, write_attn, **kwargs)
        self.canvases = None

    def record(self, do_record):
        self.canvases = [None] * T if do_record else None

    def call(self, x):
        canvas = tf.zeros((batch_size, img_size))
        h_dec = tf.zeros((batch_size, dec_size))
        enc_states = self.step.encode.get_initial_state(batch_size=batch_size,
                dtype=tf.float32)
        dec_states = self.step.decode.get_initial_state(batch_size=batch_size,
                dtype=tf.float32) 
        kl_terms = []

        for t in range(T):
            canvas, enc_states, dec_states, h_dec = \
                    self.step(x, canvas, enc_states, dec_states, h_dec)
            if isinstance(self.canvases, list):
                self.canvases[t] = canvas

        kl = tf.add_n(self.step.losses)
        Lz = tf.reduce_mean(kl)
        self.lz = Lz
        xent_terms = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=canvas)
        xent = tf.reduce_sum(xent_terms, axis=1)
        self.lx = tf.reduce_mean(xent, 0)


    def generate(self):
        canvas = tf.zeros((batch_size, img_size))
        dec_states = self.step.decode.get_initial_state(batch_size=batch_size,
                dtype=tf.float32) 
        for t in range(T):
            z = tf.random.normal((batch_size, z_size))
            canvas, dec_states = self.step.generate(canvas, z, dec_states)
            if isinstance(self.canvases, list):
                self.canvases[t] = canvas
        s = tf.math.sigmoid(canvas)
        seed = [123, 456]
        b = tf.random.stateless_binomial(s.shape, seed, [1], s) 
        return b

    def get_config(self):
        return { } 

    @classmethod
    def from_config(cls, config):
        return cls(**config)


## MODEL PARAMETERS ## 
def main(data_dir, ckpt_template=None, start_step=0, 
        ckpt_every=1000, report_every=10, 
        read_attn=True, write_attn=True,
        learning_rate=1e-3): 
    model = Draw(read_attn, write_attn)
    if start_step != 0:
        ckpt_path = ckpt_template.replace('%', str(start_step))
        model.load_weights(ckpt_path)

    @tf.autograph.experimental.do_not_convert
    def map_fn(item):
        return tf.cast(tf.reshape(item['image'], (784,)), tf.float32)

    ds = tfds.load('binarized_mnist', split='train', data_dir=data_dir)
    ds = ds.map(map_fn)
    ds = ds.shuffle(buffer_size=1024).batch(batch_size)
    ds = ds.repeat()
    dsit = iter(ds)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,
            beta_1=0.5)

    # model.compile(optimizer=opt)
    canvases = []
    Lxs = []
    Lzs = []

    for step in range(start_step, train_iters + start_step):
        x_batch_train = next(dsit)
        with tf.GradientTape() as tape:
            model(x_batch_train)
            loss = model.lx + model.lz 
            # loss = model.lz
        grads = tape.gradient(loss, model.trainable_weights)
        for gi, g in enumerate(grads):
            if g is not None:
                grads[gi] = tf.clip_by_norm(g, 5)

        if False:
            print('step: %d' % step)
            for g, v in zip(grads, model.trainable_weights):
                if g is None:
                    continue
                print('%f: %s' % (tf.math.reduce_mean(tf.math.abs(g)), v.name))
            print('\n\n')

        opt.apply_gradients(zip(grads, model.trainable_weights))
        Lxs.append(model.lx)
        Lzs.append(model.lz)

        if step % report_every == 0:
            print("iter=%d : Lx: %f Lz: %f" % (step, model.lx, model.lz))

        if step % ckpt_every == 0 and step != start_step:
            ckpt_path = ckpt_template.replace('%', str(step))
            model.save_weights(ckpt_path)
            print("Model weights saved in file: %s" % ckpt_path)

    # batch for drawing the layer reconstructions
    x_batch = next(dsit)
    model.record(True)
    model(x_batch)
    canvases = np.array(model.canvases)

    out_file=os.path.join(data_dir,"draw_data.npz")
    np.savez(out_file, canvases=canvases, Lxs=Lxs, Lzs=Lzs)
    print("Outputs saved in file: %s" % out_file)

## TRAINING FINISHED ## 


if __name__ == '__main__':
    print(sys.executable, ' '.join(arg for arg in sys.argv), file=sys.stderr,
            flush=True)
    fire.Fire(main)


