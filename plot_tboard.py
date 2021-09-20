import matplotlib.pyplot as plt
import tensorboard as tb
import pandas as pd
import fire


def main(exp_id, run):
    exp = tb.data.experimental.ExperimentFromDev(exp_id)
    df = exp.get_scalars()
    df = df.loc[df.run == run]
    mi = pd.MultiIndex.from_frame(df.loc[:, ['step', 'tag']])
    df.index = mi
    cols = df.loc[:, ['value']].unstack()

    plt.figure()
    cols.plot()
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)


