import blindspot as bs
from cslib.utils import glance
import numpy as np
import click
from pathlib import Path

@click.command()
@click.option('--dataset', default='/Volumes/Charles/data/blindpoint/source')
@click.option('--index', default=0)
@click.option('--method_list', default='default|default2')
@click.option('--need_save', default=False)
@click.option('--save_path', default='/Volumes/Charles/data/blindpoint/source')
def main(**kwargs):

    bs.set_base_path(kwargs['dataset'])
    if kwargs['index'] == 0:
        all_info = bs.get_all_proj_info()
    else:
        all_info = {kwargs['index'] : bs.get_proj_info_by_index(kwargs['index'])}
    method_list = kwargs['method_list'].split('|')

    for k, info in all_info.items():
        bs.load_low_voltages(info)
        bs.load_high_voltages(info)
        bs.load_low_noice(info)
        avg_img_h = np.average(info['img_h'], axis=0)/65536.0 # into range of [0,1]
        avg_img_l = np.average(info['img_l'], axis=0)/65536.0

        num_cols = max(5, len(method_list))

        glance(
            [avg_img_h, avg_img_l, np.abs(avg_img_h-avg_img_l), bs.norm(np.abs(info['noice_l'])), avg_img_l] +\
                [None] * (num_cols - 5) +\
                [bs.load_bad_mask(info,m) for m in method_list] + \
                [None] * (num_cols - len(method_list)),
            title = [f"{k}-High", f"{k}-Low", "abs(h-l) with auto", "Noice with auto", "3d plot vol_l"] +\
                [None] * (num_cols - 5) +\
                method_list +\
                [None] * (num_cols - len(method_list)),
            auto_contrast = [False, False, True, True, True] +\
                [None] * (num_cols - 5) +\
                [False] * len(method_list) +\
                [None] * (num_cols - len(method_list)),
            plot_3d = [False, False, False, False, True] +\
                [None] * (num_cols - 5) +\
                [False] * len(method_list) +\
                [None] * (num_cols - len(method_list)),
            shape = (2, num_cols),
            figsize = (6*num_cols, 10), 
            save = kwargs['need_save'],
            save_path = Path(kwargs['save_path']) / f"{k}.png",
        )
        bs.delete_info(info)

# f"({info['active']}) Bad:{(info['bad']==255).sum()}:{(info['bad']==0).sum()}"
if __name__ == "__main__":
    main()