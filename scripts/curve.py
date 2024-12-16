import blindspot as bs
import click

@click.command()
@click.option('--dataset', default='/Users/kimshan/Public/data/blindpoint')
def main(dataset):
    bs.BASE_PATH = dataset
    info = bs.get_proj_info_by_index(4)
    bs.pixel_voltage_response(info)
    bs.plot_3d(info['vol_response'], zlim=(-0.55, -0.40))
    bs.plot_3d(bs.curved_surface_fitting(info),zlim=(-0.55, -0.40))

    # for index,info in bs.get_all_proj_info().items():
    #     if method == "curved_surface":
    #         bad = bs.curved_surface_fitting(info,times=6)
    #         save_array_to_img(bad,Path(result) / f"{index}.png")
    #         bs.delete_info(info)

if __name__ == "__main__":
    main()