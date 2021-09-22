# All keys that appear in each entry of HPARAMS_REGISTRY must also appear in
# some entry of DEFAULTS
HPARAMS_REGISTRY = {}
DEFAULTS = {}

class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f'attribute {attr} undefined')

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


def setup_hparams(hparam_set_names, kwargs):
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(",")
    hparam_sets = [HPARAMS_REGISTRY[x.strip()] for x in hparam_set_names if x] + [kwargs]
    for k, v in DEFAULTS.items():
        H.update(v)
    for hps in hparam_sets:
        for k in hps:
            if k not in H:
                raise ValueError(f"{k} not in default args")
        H.update(**hps)
    H.update(**kwargs)
    return H


mnist = Hyperparams(
        dataset = 'binarized_mnist',
        height = 28,
        width = 28,
        enc_size = 256,
        dec_size = 256,
        num_channels = 1,
        read_n = 2,
        write_n = 5,
        z_size = 100,
        T = 64,
        batch_size = 100,
        max_steps = 100000,
        eps = 1e-8
)


cifar10 = Hyperparams(
        dataset = 'cifar10',
        height = 32,
        width = 32,
        img_size = 900,
        enc_size = 400,
        dec_size = 400,
        num_channels = 3,
        read_n = 5,
        write_n = 5,
        z_size = 200,
        T = 64,
        batch_size = 200,
        max_steps = 100000,
        eps = 1e-8
)


HPARAMS_REGISTRY["mnist"] = mnist
DEFAULTS["mnist"] = mnist

HPARAMS_REGISTRY["cifar10"] = cifar10
DEFAULTS["cifar10"] = cifar10

