#!/usr/bin/env python

"""
Adapted from Eric Jang's implementation from 2016, upgraded to TF2
"""

import os
import sys
import fire
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import layers

from hparams import setup_hparams


class Linear(layers.Dense):
    def __init__(self, output_dim):
        super(Linear, self).__init__(units=output_dim, 
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros')



class AttnWindow(tf.keras.layers.Layer):
    def __init__(self, w, h):
        super(AttnWindow, self).__init__()
        self.calcParams = Linear(5) # TODO fix this
        self.w = w
        self.h = h
        self.eps = 1e-8

    def filterbank(self, gx, gy, sigma2, delta, N):
        grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta # eq 19
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta # eq 20
        a = tf.reshape(tf.cast(tf.range(self.w), tf.float32), [1, 1, -1])
        b = tf.reshape(tf.cast(tf.range(self.h), tf.float32), [1, 1, -1])
        mu_x = tf.reshape(mu_x, [-1, N, 1])
        mu_y = tf.reshape(mu_y, [-1, N, 1])
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2))
        Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
        # normalize, sum over A and B dims
        Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keepdims=True),self.eps)
        Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keepdims=True),self.eps)
        return Fx,Fy


    def call(self, h_dec, N):
        params = self.calcParams(h_dec)
        gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(params,5,1)
        gx=(self.w+1)/2*(gx_+1)
        gy=(self.h+1)/2*(gy_+1)
        sigma2=tf.exp(log_sigma2)
        delta=(max(self.h, self.w)-1)/(N-1)*tf.exp(log_delta) # batch x N
        return self.filterbank(gx,gy,sigma2,delta,N)+(tf.exp(log_gamma),)


## READ ## 
class ReadAttn(layers.Layer):
    def __init__(self, hps, do_attn=True, **kwargs):
        super(ReadAttn, self).__init__(**kwargs)
        self.attn_window = AttnWindow(hps.width, hps.height)
        self.do_attn = do_attn
        self.w = hps.width
        self.h = hps.height
        self.nc = hps.num_channels
        self.read_n = hps.read_n

    def call(self, x, x_hat, h_dec_prev):
        # x, x_hat: bcp (batch, channel, pixel)
        bs = x.shape[0]
        if not self.do_attn:
            return tf.concat([x, x_hat], 1)

        def filter_img(img,Fx,Fy,gamma,N):
            Fxt=tf.transpose(Fx,perm=[0,1,3,2])
            img=tf.reshape(img,[-1, self.nc, self.h, self.w])
            glimpse=tf.matmul(Fy,tf.matmul(img,Fxt))
            glimpse=tf.reshape(glimpse,[-1,self.nc,N*N])
            return glimpse * tf.expand_dims(gamma, 1)

        N=self.read_n
        Fx,Fy,gamma = self.attn_window(h_dec_prev, N)
        Fx = tf.expand_dims(Fx, 1) # accommodate channels
        Fy = tf.expand_dims(Fy, 1)
        x=filter_img(x,Fx,Fy,gamma,N) # batch x (read_n*read_n)
        x_hat=filter_img(x_hat,Fx,Fy,gamma,self.read_n)
        r = tf.concat([x,x_hat], 2) # concat along feature axis
        return tf.reshape(r, [bs, -1])


## Q-SAMPLER (VARIATIONAL AUTOENCODER) ##
class SampleQ(layers.Layer):
    def __init__(self, hps, **kwargs):
        super(SampleQ, self).__init__(**kwargs)
        self.mu = Linear(hps.z_size)
        self.logsigma = Linear(hps.z_size)

    def call(self, h_enc, noise):
        _mu = self.mu(h_enc)
        _logsigma = self.logsigma(h_enc)
        _sigma = tf.exp(_logsigma)
        return (_mu + _sigma * noise, _mu, _logsigma, _sigma)


## WRITER ## 
class WriteAttn(layers.Layer):
    def __init__(self, hps, do_attn=True, **kwargs):
        super(WriteAttn, self).__init__(**kwargs)
        if do_attn:
            write_size = hps.write_n*hps.write_n
        else:
            write_size = hps.height * hps.width

        self.calcW = Linear(write_size * hps.num_channels)
        self.attn_window = AttnWindow(hps.width, hps.height)
        self.do_attn = do_attn
        self.hps = hps

    def call(self, h_dec):
        if not self.do_attn:
            return self.calcW(h_dec)

        N=self.hps.write_n
        w = self.calcW(h_dec)
        w=tf.reshape(w,[self.hps.batch_size,self.hps.num_channels,N,N])
        Fx,Fy,gamma=self.attn_window(h_dec, N)
        Fx = tf.expand_dims(Fx, 1) # accommodate channels
        Fy = tf.expand_dims(Fy, 1)
        Fyt=tf.transpose(Fy,perm=[0,1,3,2])
        wr=tf.matmul(Fyt,tf.matmul(w,Fx))
        shape = [self.hps.batch_size, self.hps.num_channels, 
                self.hps.width * self.hps.height]
        wr=tf.reshape(wr, shape)
        #gamma=tf.tile(gamma,[1,B*A])
        return wr * tf.expand_dims(1.0 / gamma, 1)



class TimeStep(layers.Layer):
    def __init__(self, hps, read_attn=True, write_attn=True, **kwargs):
        super(TimeStep, self).__init__(**kwargs)
        self.read = ReadAttn(hps, read_attn, name='read') 
        self.encode = layers.LSTMCell(hps.enc_size, name='encoder')
        self.sample = SampleQ(hps, name='sample')
        self.decode = layers.LSTMCell(hps.dec_size, name='decoder')
        self.write = WriteAttn(hps, write_attn, name='write')
        self.z_size = hps.z_size

    def call(self, x, canvas, enc_states, dec_states, h_dec):
        bs = x.shape[0]
        x_hat = x - tf.sigmoid(canvas)
        r = self.read(x, x_hat, h_dec)
        enc_inputs = tf.concat([r, h_dec], 1)
        h_enc, enc_states = self.encode(enc_inputs, enc_states)
        noise = tf.random.normal((bs, self.z_size), mean=0, stddev=1) 
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



class Draw(tf.keras.Model):
    def __init__(self, hps, read_attn=True, write_attn=True, **kwargs):
        super(Draw, self).__init__(**kwargs)
        self.step = TimeStep(hps, read_attn, write_attn, **kwargs)
        self.canvases = None
        self.bs = hps.batch_size
        self.nc = hps.num_channels
        self.isz = hps.width * hps.height
        self.zsz = hps.z_size
        self.dsz = hps.dec_size
        self.T = hps.T

    def record(self, do_record):
        self.canvases = [None] * self.T if do_record else None

    def call(self, x):
        canvas = tf.zeros((self.bs, self.nc, self.isz))
        h_dec = tf.zeros((self.bs, self.dsz))
        enc_states = self.step.encode.get_initial_state(batch_size=self.bs, dtype=tf.float32)
        dec_states = self.step.decode.get_initial_state(batch_size=self.bs, dtype=tf.float32) 
        kl_terms = []

        for t in range(self.T):
            canvas, enc_states, dec_states, h_dec = \
                    self.step(x, canvas, enc_states, dec_states, h_dec)
            if isinstance(self.canvases, list):
                self.canvases[t] = canvas

        kl = tf.add_n(self.step.losses)
        Lz = tf.reduce_mean(kl)
        self.lz = Lz
        xent_terms = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=canvas)
        xent = tf.reduce_sum(xent_terms, axis=[1,2])
        self.lx = tf.reduce_mean(xent, 0)


    def generate(self):
        canvas = tf.zeros((self.bs, self.nc, self.isz))
        dec_states = self.step.decode.get_initial_state(batch_size=self.bs,
                dtype=tf.float32) 
        for t in range(self.T):
            z = tf.random.normal((self.bs, self.zsz))
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
def main(dataset, data_dir, hps, tboard_logdir, 
        ckpt_template=None, start_step=0, 
        ckpt_every=1000, report_every=10, 
        read_attn=True, write_attn=True,
        learning_rate=1e-3,
        max_steps=10000, **kwargs): 

    hps = setup_hparams(hps, kwargs)
    model = Draw(hps, read_attn, write_attn)
    if start_step != 0:
        ckpt_path = ckpt_template.replace('%', str(start_step))
        model.load_weights(ckpt_path)

    @tf.autograph.experimental.do_not_convert
    def mnist_map_fn(item):
        img = item['image']
        img = tf.reshape(img, (1, 784))
        img = tf.cast(img, tf.float32)
        return img

    def cifar_map_fn(item):
        img = item['image']
        img = tf.transpose(img, [2, 0, 1]) # move the channels dim in front
        img = tf.reshape(img, (3, 1024))
        img = tf.cast(img, tf.float32)
        img = img / 255.0
        return img

    if dataset == 'mnist':
        ds = tfds.load('binarized_mnist', split='train', data_dir=data_dir)
        ds = ds.map(mnist_map_fn)

    elif dataset == 'cifar10':
        ds = tfds.load('cifar10', split='train', data_dir=data_dir)
        ds = ds.map(cifar_map_fn)

    ds = ds.shuffle(buffer_size=1024).batch(hps.batch_size)
    ds = ds.repeat()
    dsit = iter(ds)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate,
            beta_1=0.5)

    # model.compile(optimizer=opt)
    writer = tf.summary.create_file_writer(tboard_logdir)

    for step in range(start_step, max_steps + start_step):
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

        if step % report_every == 0:
            print("iter=%d : Lx: %f Lz: %f" % (step, model.lx, model.lz))
            tf.summary.flush(writer)

        if step % ckpt_every == 0 and step != start_step:
            ckpt_path = ckpt_template.replace('%', str(step))
            model.save_weights(ckpt_path)
            print("Model weights saved in file: %s" % ckpt_path)

        with writer.as_default():
            tf.summary.scalar('loss/lx', model.lx, step=step)
            tf.summary.scalar('loss/lz', model.lz, step=step)
            tf.summary.scalar('loss/total', loss, step=step)


if __name__ == '__main__':
    print(sys.executable, ' '.join(arg for arg in sys.argv), file=sys.stderr,
            flush=True)
    fire.Fire(main)


