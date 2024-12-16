import blindspot.convert as convert
import click
import csv
from pathlib import Path
from PIL import Image

@click.command()
@click.option('--src', default='/Users/kimshan/Public/data/test')
@click.option('--dest', default='/Users/kimshan/Public/data/blindpoint')
@click.option('--only_csv', default=False)
def main(src,dest,only_csv):
    convert.BASE_PATH = Path(src)
    pathinfo_csv_path = Path(dest) / 'pathinfo.csv'

    if Path(dest).exists() == False:
        Path(dest).mkdir()
    count = 0
    
    # 如果文件存在，读取现有内容
    existing_paths = set()
    if pathinfo_csv_path.exists():
        with open(pathinfo_csv_path, 'r', newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            existing_paths = {Path(row['path']):row['index'] for row in csv_reader}
    else:
        with open(pathinfo_csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['index', 'width', 'height', 'temp_l', 'temp_h', 'num_l', 'num_h', 'scale', 'path', 'active'])
    
    # 打开文件准备写入
    with open(pathinfo_csv_path, 'a', newline='') as csv_file:
        count = len(existing_paths)
        csv_writer = csv.writer(csv_file)
        for path in convert.get_src_list():
            info = convert.get_src_info(path)
            if info == False:
                continue # 目录不合法
            if path in existing_paths:
                print(f'Existed {existing_paths[path]}: {path}')
                continue  # 如果存在，跳过
            count += 1
            csv_writer.writerow([
                count, info['width'], info['height'], info['temp_l'], info['temp_h'], 
                info['num_l'], info['num_h'], info['scale'], path, True
            ])
            if (Path(dest) / f'{count}').exists() == False:
                (Path(dest) / f'{count}').mkdir()
                (Path(dest) / f'{count}' / f'{info["temp_l"]}').mkdir()
                (Path(dest) / f'{count}' / f'{info["temp_h"]}').mkdir()
            if only_csv == False:
                for i,img in enumerate(info[f'img_l']):
                    Image.fromarray(img).save(str(Path(dest) / f'{count}' / f'{info["temp_l"]}' / f'{i:03d}.png'), format='png')
                for i,img in enumerate(info[f'img_h']):
                    Image.fromarray(img).save(str(Path(dest) / f'{count}' / f'{info["temp_h"]}' / f'{i:03d}.png'), format='png')
            
            print(f"{count}: {path}")

if __name__ == "__main__":
    main()
