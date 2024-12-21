import blindspot as bs
import click
from pathlib import Path
import numpy as np
from clib.utils import glance,save_array_to_img

@click.command()
@click.option('--dataset', default='/Users/kimshan/Public/data/blindpoint')
@click.option('--index', default=0)
@click.option('--method', default='curved_surface')
@click.option('--need_save', default=False)
@click.option('--result', default='/path/to/result/folder')
@click.option('--curve_sigma', default=3)
def main(**kwargs):
    bs.BASE_PATH = kwargs['dataset']
    if kwargs['index'] == 0:
        all_info = bs.get_all_proj_info()
    else:
        all_info = {kwargs['index'] : bs.get_proj_info_by_index(kwargs['index'])}
    for index,info in all_info.items():
        if kwargs['method'] == "curved_surface":
            bad = bs.dect_curved_surface_fitting(info,times=kwargs['curve_sigma'])
        elif kwargs['method'] == 'gb':
            bad = bs.dect_gb(info)
        else:
            raise ValueError("Unexpected detect method")
        
        if kwargs['need_save']:
            save_array_to_img(bad,Path(kwargs['result']) / f"{index}.png")
        else:
            glance(
                image=[np.average(info['img_l'],axis=0),bad,np.average(info['img_l'],axis=0)],
                title=["img L", f"Bad Mask {kwargs['method']}", "3D plot"],
                plot_3d=[False,False,True]
            )
        bs.delete_info(info)

if __name__ == "__main__":
    main()
