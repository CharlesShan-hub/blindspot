import blindspot as bs
from clib.utils import glance
import numpy as np
import click
from pathlib import Path

@click.command()
@click.option('--dataset', default='/Users/kimshan/Public/data/blindpoint')
@click.option('--index', default=0)
@click.option('--method_list', default='default|default2')
@click.option('--need_save', default=False)
@click.option('--save_path', default='/Users/kimshan/Public/data/blindpoint')
def main(**kwargs):

    bs.BASE_PATH = kwargs['dataset']
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

        glance(
            image = [avg_img_h, avg_img_l, np.abs(avg_img_h-avg_img_l), bs.norm(np.abs(info['noice_l'])), avg_img_l] +\
                [None] * ((len(method_list)-5) if len(method_list) > 5 else 0) +\
                [bs.load_bad_mask(info,m) for m in method_list] + \
                [None] * ((5-len(method_list)) if len(method_list) <=5 else 0),
            title = [f"{k}-High", f"{k}-Low", "abs(h-l) with auto", "Noice with auto", "3d plot vol_l"] +\
                [None] * ((len(method_list)-5) if len(method_list) > 5 else 0) +\
                method_list +\
                [None] * ((5-len(method_list)) if len(method_list) <= 5 else 0),
            auto_contrast = [False, False, True, True, True] +\
                [None] * ((len(method_list)-5) if len(method_list) > 5 else 0) +\
                [False] * len(method_list) +\
                [None] * ((5-len(method_list)) if len(method_list) <= 5 else 0),
            plot_3d = [False, False, False, False, True] +\
                [None] * ((len(method_list)-5) if len(method_list) > 5 else 0) +\
                [False] * len(method_list) +\
                [None] * ((5-len(method_list)) if len(method_list) <= 5 else 0),
            shape = (2,len(method_list) if len(method_list) > 2 else 2),
            figsize = (20*(len(method_list) if len(method_list) > 2 else 2), 16*2), 
            save = kwargs['need_save'],
            save_path = Path(kwargs['save_path']) / f"{k}.png",
        )
        bs.delete_info(info)

# f"({info['active']}) Bad:{(info['bad']==255).sum()}:{(info['bad']==0).sum()}"
if __name__ == "__main__":
    main()