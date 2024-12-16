import blindspot as bs
import click
from pathlib import Path
from clib.utils import glance,save_array_to_img

@click.command()
@click.option('--dataset', default='/Users/kimshan/Public/data/blindpoint')
@click.option('--method', default='curved_surface')
@click.option('--result', default='/path/to/result/folder')
def main(dataset,method,result):
    bs.BASE_PATH = dataset
    # info = bs.get_proj_info_by_index(4)
    for index,info in bs.get_all_proj_info().items():
        if method == "curved_surface":
            bad = bs.curved_surface_fitting(info,times=6)
            save_array_to_img(bad,Path(result) / f"{index}.png")
            bs.delete_info(info)

if __name__ == "__main__":
    main()
