import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
from clib.utils import to_image
import blindspot as bs
import click

CONFIG_WIDTH = 10
TITLE_FONT = ("Arial", 24)
CONTENT_FONT = ("Arial", 15)

class App:
    def __init__(self, root, **kwargs):
        # Setting
        self.root = root
        self.root.title("Charles App")
        self.root.geometry(f"{kwargs['window_width']}x{kwargs['window_height']}")
        self.base_src = bs.BASE_PATH = Path(kwargs['base_src'])
        self.save_dir = Path(kwargs["save_dir"])
        self.double_temp = kwargs['double_temp']
        self.r = kwargs['r']
        self.ui_frames()
        self.ui_config()
        self.ui_pixel_choose_widgets()
        self.ui_result_show_widgets()
        self.proj_conf_comb.current(4)
        self.select_proj(None)
    
    def ui_frames(self):
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.pic_frame = tk.Frame(self.bottom_frame)
        self.pic_frame.pack(side=tk.LEFT)
        self.res_frame = tk.Frame(self.bottom_frame)
        self.res_frame.pack(side=tk.LEFT)
    
    def ui_config(self):
        # Label
        self.text_label = tk.Label(self.top_frame, text="Blind Pixels Manual Search", font=TITLE_FONT, pady=15)
        self.text_label.pack(side=tk.TOP, fill=tk.X)

        # Base Path Config
        self.base_src_frame = tk.Frame(self.top_frame)
        self.base_src_label = tk.Label(self.base_src_frame, width=CONFIG_WIDTH, text='Base Src', font=CONTENT_FONT)
        self.base_src_entry = tk.Entry(self.base_src_frame, font=CONTENT_FONT)
        self.base_src_entry.insert(0,self.base_src)
        self.base_src_button = tk.Button(self.base_src_frame, text="Choose", command=self.select_base_directory, font=CONTENT_FONT)
        self.base_src_frame.pack(side=tk.TOP, fill=tk.X)
        self.base_src_label.pack(side=tk.LEFT, padx=10)
        self.base_src_entry.pack(side=tk.LEFT, fill=tk.X,expand=True)
        self.base_src_button.pack(side=tk.LEFT)

        # Proj chooser
        self.proj_list = self.init_proj_conf_list()
        self.proj_conf_frame = tk.Frame(self.top_frame)
        self.proj_conf_label = tk.Label(self.top_frame, width=CONFIG_WIDTH, text='Project', font=CONTENT_FONT)
        self.proj_conf_comb = ttk.Combobox(self.top_frame, values=self.proj_list, font=CONTENT_FONT)
        self.proj_conf_comb.bind("<<ComboboxSelected>>", self.select_proj)
        self.proj_conf_frame.pack(side=tk.TOP,fill=tk.X)
        self.proj_conf_label.pack(side=tk.LEFT,padx=10)
        self.proj_conf_comb.pack(side=tk.LEFT,fill=tk.X,expand=True)

    def ui_pixel_choose_widgets(self):
        # Pixel Choose
        self.zoom_factor = 1

        self.label_pre_info = tk.Label(self.pic_frame, width=CONFIG_WIDTH, text='Please Choose Pixel', font=CONTENT_FONT)
        self.label_pre_info.pack(side=tk.TOP, fill=tk.X)

        self.canvas_scro_frame = tk.Frame(self.pic_frame)
        self.canvas = tk.Canvas(self.canvas_scro_frame, width=450, height=600, bg='#fff', scrollregion=(0,0,10,10))
        self.scrollY = tk.Scrollbar(self.canvas_scro_frame, command=self.canvas.yview, orient='vertical')
        self.canvas.config(yscrollcommand=self.scrollY.set)
        self.scrollX = tk.Scrollbar(self.pic_frame, command=self.canvas.xview, orient='horizontal')
        self.canvas.config(xscrollcommand=self.scrollX.set)
        self.canvas.bind("<Button-1>", self.select_pixel)
        self.scrollX.pack(side=tk.TOP, fill=tk.X)
        self.canvas_scro_frame.pack(side=tk.TOP)
        self.canvas.pack(side=tk.LEFT)
        self.scrollY.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_tool_frame = tk.Frame(self.pic_frame)
        self.canvas_tool_frame.pack(side=tk.TOP, expand=True)
        self.zoom_button = tk.Button(self.canvas_tool_frame, text="Zoom In", command=self.zoom_in_image, font=CONTENT_FONT)
        self.zoom_button.pack(side=tk.LEFT)
        self.zoom_button = tk.Button(self.canvas_tool_frame, text="Zoom Out", command=self.zoom_out_image, font=CONTENT_FONT)
        self.zoom_button.pack(side=tk.LEFT)
        self.draw_wave_button = tk.Button(self.canvas_tool_frame, text="Draw", font=CONTENT_FONT, command=self.draw_wave)
        self.draw_wave_button.pack(side=tk.LEFT)

    def ui_result_show_widgets(self):
        self.label_res = tk.Label(self.res_frame, width=CONFIG_WIDTH, text='', font=CONTENT_FONT)
        self.label_res.pack(side=tk.TOP,expand=True,fill=tk.X)
        self.canvas_res = tk.Canvas(self.res_frame, width=700, height=650, bg='#fff')
        self.canvas_res.pack(side=tk.LEFT)
    
    def zoom_in_image(self):
        self.zoom_factor = self.zoom_factor * 2  # 假设放大倍数为2
        self.temp = self.noice.copy()
        self.noice = np.zeros(shape=np.array(self.noice.shape)*2)
        self.noice = np.repeat(np.repeat(self.temp, 2, axis=0), 2, axis=1)
        self.zoomed_photo_image = ImageTk.PhotoImage(to_image(bs.norm(self.noice)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.zoomed_photo_image)
        self.canvas.config(scrollregion=(0,0,self.noice.shape[1],self.noice.shape[0]))
        self.canvas.xview_moveto(self.canvas.xview()[0] * 2)
        self.canvas.yview_moveto(self.canvas.yview()[0] * 2)

    def zoom_out_image(self):
        if self.zoom_factor < 2:
            return
        self.zoom_factor = int(self.zoom_factor / 2)  # 假设放大倍数为2
        self.temp = self.noice.copy()
        self.noice = np.zeros(shape=(np.array(self.temp.shape)/2).astype('int64'))
        self.noice = self.temp[::2, ::2]
        self.zoomed_photo_image = ImageTk.PhotoImage(to_image(bs.norm(self.noice)))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.zoomed_photo_image)
        self.canvas.config(scrollregion=(0,0,self.noice.shape[1],self.noice.shape[0]))
        self.canvas.xview_moveto(self.canvas.xview()[0] / 2)
        self.canvas.yview_moveto(self.canvas.yview()[0] / 2)
    
    def select_pixel(self,event):
        # 这里可以添加选择像素的逻辑
        # 例如，你可以获取鼠标点击的坐标，并将其转换为像素值
        x, y = self.canvas.winfo_pointerx(), self.canvas.winfo_pointery()
        x = (x - self.canvas.winfo_rootx()) - self.canvas.winfo_x()
        y = (y - self.canvas.winfo_rooty()) - self.canvas.winfo_y()
        x, y = self.canvas.canvasx(x), self.canvas.canvasy(y)
        x, y = int(x/self.zoom_factor), int(y/self.zoom_factor)
        # 这里可以添加选择像素的逻辑，例如打印像素值
        # print(f"Selected pixel: (y={y}, x={x})")
        self.label_pre_info.config(text=f'h = {y}, w = {x} | is_bad:{self.bad[y,x]==255}')
        self.point = (y,x)

        # 框选出选中的位置
        range_size = 21
        if self.zoom_factor > 7:
            rh = (y-int(range_size/2)) * self.zoom_factor
            rw = (x-int(range_size/2)) * self.zoom_factor
            if hasattr(self, 'selected_pixel_rect'):
                for item in self.selected_pixel_rect:
                    self.canvas.delete(item)
            self.selected_pixel_rect = []
            for i in range(range_size):
                for j in range(range_size):
                    try:
                        self.selected_pixel_rect.append(
                            self.canvas.create_rectangle(
                                rw + i*self.zoom_factor, 
                                rh + j*self.zoom_factor, 
                                rw + (i+1)*self.zoom_factor-1,
                                rh + (j+1)*self.zoom_factor-1,
                                outline='red' if self.bad[y-int(range_size/2)+j,x-int(range_size/2)+i]==255 else 'green', 
                                width=1
                            )
                        )
                    except:
                        pass
    
    def select_base_directory(self):
        self.base_src = Path(filedialog.askdirectory())
        self.base_src_entry.delete(0)
        self.base_src_entry.insert(0,self.base_src)
        bs.BASE_PATH = self.base_src
    
    def select_proj(self,event):
        [self.proj_id,self.path] = self.proj_conf_comb.get().split(':')
        self.path = Path(self.path)
        assert self.path.exists()
        info = bs.get_proj_info_by_index(self.proj_id)
        bs.load_low_noice(info)
        bs.load_high_noice(info)
        bs.load_high_voltages(info)
        bs.load_low_voltages(info)
        self.shape = (info['width'],info['height'])
        self.noice = info['noice_l']
        self.noice_35 = info['noice_h']
        self.voltage = info['vol_l']
        self.voltage_35 = info['vol_h']
        self.bad = bs.read_png_to_array(Path(info['path']) / 'BadPixel.png')
        
        self.photo_image = ImageTk.PhotoImage(to_image(bs.norm(self.noice)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
        self.canvas.config(scrollregion=(0,0,self.noice.shape[1],self.noice.shape[0]))
        self.zoom_factor = 1

        self.average_noice = np.average(self.noice)
        self.average_noice_35 = np.average(self.noice_35)
        self.average_image = np.average(self.voltage,axis=0)
        self.average_image_35 = np.average(self.voltage_35,axis=0)
        self.label_pre_info.config(text='Chooes Pixcel First')
    
    def init_proj_conf_list(self):
        if self.base_src.exists() == False:
            return []
        temp = []
        for _,info in bs.get_all_proj_info().items():
            temp.append(f'{info["index"]}:{info["path"]}')
        return temp
    
    def draw_wave(self):
        bs.plot_wave(
            p = self.point,                     # 中心点坐标
            r = self.r,                         # 中心点周围扩展像素个数
            bad = self.bad,                     # 盲元表
            proj_id = self.proj_id,             # 实验 ID
            save_dir = self.save_dir,           # 保存目录
            double_temp = self.double_temp,     # 绘制两个温度曲线
            vol_l= self.voltage,                # 低温电压
            vol_h= self.voltage_35,             # 高温电压
            vol_l_avg = self.average_image,     # 低温平均电压
            vol_h_avg = self.average_image_35,  # 低温平均电压
            noice_l_avg = self.average_noice,   # 平均低温噪声
            noice_h_avg = self.average_noice_35 # 平均低温噪声
        )
        self.res_image = ImageTk.PhotoImage(Image.open(self.save_dir / f"{self.proj_id}" / f"{self.point[0]}_{self.point[1]}.png").resize((700,650)))
        self.canvas_res.create_image(0, 0, anchor=tk.NW, image=self.res_image)
        self.label_res.config(text=f'proj: {self.proj_id}, h: {self.point[0]}, w: {self.point[1]}')
    

@click.command()
@click.option('--base_src', default='/Users/kimshan/Public/data/blindpoint')
@click.option('--save_dir', default='/Users/kimshan/Public/data/blindpoint/temp')
@click.option('--r', type=int, default=3)
@click.option('--double_temp', type=bool, default=True)
@click.option('--window_width', type=int, default=1200)
@click.option('--window_height', type=int, default=820)
def main(**kwargs):
    root = tk.Tk()
    App(root,**kwargs)
    root.mainloop()
    

if __name__ == "__main__":
    main()
