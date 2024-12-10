import blindspot as bs
import click
import numpy as np
from clib.utils import glance

@click.command()
@click.option('--dataset', default='/Users/kimshan/Public/data/blindpoint')
def main(dataset):
    bs.BASE_PATH = dataset
    for index,info in bs.get_all_proj_info().items():
        bs.load_high_voltages(info)
        bs.load_low_voltages(info)
        glance([np.average(info['vol_l'],axis=0),np.average(info['vol_h'],axis=0)])
        user_input = input(f"Continue with this {index} data? (y/n): ").strip().lower()
        info['active'] = True if user_input != 'n' else False 
        bs.change_info(info)
        bs.delete_info(info)

if __name__ == "__main__":
    main()
