import blindspot as bs
import click

@click.command()
@click.option('--dataset', default='/Users/kimshan/Public/data/blindpoint')
def main(dataset):
    bs.BASE_PATH = dataset
    info = bs.get_proj_info_by_index(4)
    bs.pixel_voltage_response(info)
    bs.plot_3d(info['vol_response'])#, zlim=(0.15, 0.4)
    bs.plot_3d(bs.curved_surface_fitting(info))
    
if __name__ == "__main__":
    main()