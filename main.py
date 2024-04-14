import cv2
import numpy as np
import sys, os, glob, numpy
from skimage import io
from PIL import Image, ImageTk
import tkinter as tk
import time
from tkinter import ttk
from tkinter import IntVar
import xlrd
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import concurrent.futures
import math
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess

from PIL import Image, ImageDraw
import cv2
import numpy as np
import onnxruntime
import pymysql
from tkinter import messagebox
import threading
class ImageGallery:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.images = []
        self.image_names = []  # 存储图片文件名
        self.current_index = 0

        # 加载指定文件夹中的所有图片
        self.load_images()

    def load_images(self):
        # 获取文件夹中的所有图片文件
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 加载图片并添加到图片列表中
        for file in image_files:
            image_path = os.path.join(self.image_folder, file)
            image = cv2.imread(image_path)
            self.images.append(image)
            self.image_names.append(file)  # 存储图片文件名

        # 显示第一张图片
        self.show_image(0)

    def show_image(self, index):
        # 根据索引显示指定位置的图片
        image = self.images[index]
        image_name = self.image_names[index]

        # 在图片上绘制照片名
        cv2.putText(image, image_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Image", image)
        cv2.waitKey(0)

        # 更新当前索引
        self.current_index = index

    def show_previous_image(self):
        # 显示前一张图片
        if self.current_index > 0:
            self.show_image(self.current_index - 1)

    def show_next_image(self):
        # 显示下一张图片
        if self.current_index < len(self.images) - 1:
            self.show_image(self.current_index + 1)

def person_sum(in_video,out_photo):
# 视频文件路径
    #path = '数据'+'/'+str(class_room_chosen.get()) + str(course_time_chosen.get())
    #pic_path = str(class_room_chosen.get()) + str(course_time_chosen.get()) + '.mp4'
    #in_put = in_video
    #out_put = path+'/'+'photo'
    in_put = in_video

    # 输出图片的目录
    output_dir = out_photo

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(in_put)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每分钟的帧数
    frames_per_minute = int(fps * 60)

    # 跳过前两分钟
    for _ in range(2 * frames_per_minute):
        cap.read()

    # 对于第3到第8分钟，每分钟保存一帧
    for minute in range(3, 9):
        # 读取并保存帧
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f'{minute}_minutes.jpg'), frame)
        # 跳过剩余的帧
        for _ in range(frames_per_minute - 1):
            cap.read()

    # 释放视频文件
    cap.release()


def generate_date(fload_path,dialog):
    scan(fload_path + '/' + 'class_video.mp4',fload_path + '/' + 'photo')
    person_sum(fload_path + '/' + 'class_video.mp4',fload_path + '/' + 'human_photo')
    human_model = Model('./Model/human.onnx', './Model/head.onnx', count_human=True)
    head_up_model = Model('./Model/human.onnx', './Model/head.onnx',  fload_path + '/' + 'head_up_photo',
                          count_human=False)

    num_list = []
    human_files = os.listdir(fload_path + '/' + 'human_photo')
    for file in human_files:
        file_path = fload_path+'/human_photo/'+file
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        num = human_model.model(image)
        num_list.append(num)
    human_num = math.ceil(sum(num_list) / len(num_list))

    df = pd.DataFrame(columns=['time', 'value'])
    # 获取文件夹中的文件名
    folder_path = fload_path + '/' + 'photo'
    files = os.listdir(folder_path)

    # 将文件名中的数字部分提取出来并排序
    files = sorted(files, key=lambda x: int(x.split('_')[0]))
    for file in files:
        my_file = os.path.join(folder_path, file)
        ######################
        file_path =fload_path+'/photo/'+my_file
        image = cv2.imread(os.path.join(folder_path, file))
        filename = os.path.basename(os.path.join(folder_path, file))
        head_num = head_up_model.model(image)['number']
        output_image = head_up_model.model(image)['image']
        image_out=str(fload_path + '/head_up_photo/'+filename)
        cv2.imwrite(image_out,output_image)
        value = head_num / human_num
        value = round(float(value),2)
        x=my_file[28:-12]
        df = df.append({'time': x, 'value': value,}, ignore_index=True)
    df = df.append({'person_num': human_num }, ignore_index=True)
        # 保存 DataFrame 到 Excel 文件
    df.to_excel(fload_path+'/'+'result.xlsx', index=False)
    dialog.destroy()
def analyze_data():
    dialog = tk.Toplevel()
    dialog.title("数据分析中")
    dialog.geometry("200x100")
    dialog.transient(window)
    dialog.grab_set()

    # 禁用父窗口上的所有控件
    for child in window.winfo_children():
        try:child.configure(state='disabled')
        except:pass

    generate_date('./data'+'/'+str(class_room_chosen.get()) + str(course_time_chosen.get()), dialog)

    # 恢复父窗口上的控件状态
    for child in window.winfo_children():
        try:child.configure(state='normal')
        except:pass


class Model:
    def __init__(self, model_path_human, model_path_head, output_path=None, count_human=None):
        self.model_human = model_path_human
        self.model_head = model_path_head
        self.output_path = output_path
        self.count_human = count_human

    def Getdata(self, model, input_image):
        model_path = model
        session = onnxruntime.InferenceSession(model_path)
        image = input_image
        resized_image = cv2.resize(image, (640, 640))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_data = resized_image_rgb.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        output = session.run(None, {'images': input_data})
        return output

    def draw_rectangle(self,input_image, local):
        image = input_image
        original_size = image.shape[:2]
        resized_image = cv2.resize(image, (640, 640))
        PIL_image=Image.fromarray(resized_image )
        draw = ImageDraw.Draw(PIL_image)
        for point in local:
            draw.rectangle(point, outline="red")
        finial_image=np.array(PIL_image)
        resized_image=cv2.resize(finial_image,tuple(reversed(original_size)))
        return resized_image
#
    def model(self, input_image):
        if self.count_human:
            data = self.Getdata(self.model_human, input_image)
            number = data[0]

            return number
        else:
            data = self.Getdata(self.model_head, input_image)
            number = data[0]
            output_image = self.draw_rectangle(input_image, data[1][0])
            structure = {
                'image': output_image,
                'number': number}
            return structure


    def video_show(self,video_path,input_time):
        if input_time=='':
            time=0
        else:
            time=int(input_time)
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_time = int(time * 60)
        start_frame = int(start_time*fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            else:
                results = self.model(frame)
                annotated_frame = results['image']
                cv2.imshow("video-show", annotated_frame)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if frame_count >= total_frames:
                    break
        cap.release()
        cv2.destroyAllWindows()


    def get_location(self):
        return self.output_path

def show_my_video(filepath,time):
    head_up_model = Model('./Model/human.onnx', './Model/head.onnx',  count_human=False)
    head_up_model.video_show(filepath,time)

def show_high_class_content(data_file,video_file):


    video_file = video_file

    with open(data_file+'/high.txt', 'r') as file:
        timestamps = file.read().split(', ')

    high_output_folder=data_file+'/high_rate_time'
    if not os.path.exists(high_output_folder):
        os.makedirs(high_output_folder)


    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    for index, timestamp in enumerate(timestamps):
# 转换时间戳为浮点数，乘以60转换为秒数
        time=timestamp
        timestamp = int(float(timestamp)) * 60

# 计算对应的帧数
        frame_number = int(timestamp * fps)

# 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# 读取帧
        success, frame = cap.read()

        if success:
    # 构造输出文件名
            high_output_file = f'{high_output_folder}/{time}_minutes.jpg'

    # 保存截图
            cv2.imwrite(high_output_file, frame)

    cap.release()
    gallery = ImageGallery(image_folder=high_output_folder)


    while True:
        key = cv2.waitKey(1) & 0xFF

        # 按下 "q" 键退出展示循环
        if key == ord('q'):
            break

        # 按下 "p" 键显示上一张图片
        if key == ord('a'):
            gallery.show_previous_image()

        # 按下 "n" 键显示下一张图片
        if key == ord('d'):
            gallery.show_next_image()
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    # 关闭展示窗口
    cv2.destroyAllWindows()
    

def show_low_class_content(data_file,video_file):


    video_file = video_file

    with open(data_file+'/low.txt', 'r') as file:
        timestamps = file.read().split(', ')

    low_output_folder=data_file+'/low_rate_time'
    if not os.path.exists(low_output_folder):
        os.makedirs(low_output_folder)


    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    for index, timestamp in enumerate(timestamps):
# 转换时间戳为浮点数，乘以60转换为秒数
        time=timestamp
        timestamp = int(float(timestamp)) * 60

# 计算对应的帧数
        frame_number = int(timestamp * fps)

# 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# 读取帧
        success, frame = cap.read()

        if success:
    # 构造输出文件名
            low_output_file = f'{low_output_folder}/{time}_minutes.jpg'

    # 保存截图
            cv2.imwrite(low_output_file, frame)

    cap.release()
    gallery = ImageGallery(image_folder=low_output_folder)


    while True:
        key = cv2.waitKey(1) & 0xFF

        # 按下 "q" 键退出展示循环
        if key == ord('q'):
            break

        # 按下 "p" 键显示上一张图片
        if key == ord('a'):
            gallery.show_previous_image()

        # 按下 "n" 键显示下一张图片
        if key == ord('d'):
            gallery.show_next_image()
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    # 关闭展示窗口
    cv2.destroyAllWindows()
    



#绘图
def draw(path):
    # 读取Excel数据
    df = pd.read_excel(path)

    # 创建一个新的Tkinter窗口
    new_window = tk.Tk()

    # 创建一个新的matplotlib图形
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 6))  # 更改图形大小

    # 绘制折线图，假设我们有一个名为'time'的时间列和一个名为'value'的值列
    df['time'] = df['time'].astype(str)
    df['value'] = df['value'].astype(float)
    ax.plot(df['time'], df['value'])

    # 添加坐标轴标签
    ax.set_xlabel('Time')
    ax.set_ylabel('value')

    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 5 == 0:  # 这里的5是你想要的稀疏度
            label.set_visible(True)
        else:
            label.set_visible(False)
    # 将matplotlib图形添加到Tkinter窗口中
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # 运行Tkinter事件循环
    new_window.mainloop()

#多线程截取照片
def scan(in_video,out_photo):

    video_path = in_video

    # 输出图片的目录
    output_dir = out_photo

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算每60秒的帧数
    frames_per_60s = int(fps * 60)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算需要截取的帧的数量
    num_frames_to_extract = math.ceil(total_frames / frames_per_60s)

    # 先在主线程中提取所有需要的帧并保存到磁盘
    for i in range(num_frames_to_extract):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frames_per_60s)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f'{i}_minutes.jpg'), frame)

    # 释放视频文件
    cap.release()
def get_in():
    # GUI代码
    #root.destroy()
    window = tk.Tk()  # 这是一个窗口object
    window.title('抬头率监测系统')
    window.geometry('800x600')  # 窗口大小
    #print(my_model.get())
    def read_data():
        path = r'py_excel.xls'
        data = xlrd.open_workbook(path)
        # 根据sheet名称获取
        sheet1 = data.sheet_by_name('Sheet1')
        sheet2 = data.sheet_by_name('Sheet2')
        # 获取sheet（工作表）行（row）、列（col）数
        nrows = sheet1.nrows  # 行
        ncols = sheet1.ncols  # 列
        # print(nrows, ncols)

        # 获取教室名称列表
        global room_name, time_name
        room_name = sheet2.col_values(0)
        time_name = sheet2.col_values(1)
        print(room_name)
        print(time_name)
        # 获取单元格数据
        # 1.cell（单元格）获取
        cell_A1 = sheet2.cell(0, 0).value
        print(cell_A1)
        # 2.使用行列索引
        cell_A2 = sheet2.row(0)[1].value

    read_data()

    def gettime():  # 当前时间显示
        timestr = time.strftime('%Y.%m.%d %H:%M', time.localtime(time.time()))
        lb.configure(text=timestr)
        window.after(1000, gettime)

    lb = tk.Label(window, text='', font=("黑体", 20))
    lb.grid(column=0, row=0)
    gettime()

    # 选择教室标签加下拉菜单
    choose_classroom = tk.Label(window, text="选择教室", width=15, height=2, font=("黑体", 12)).grid(column=0, row=1,sticky='w')
    class_room = tk.StringVar()
    global class_room_chosen
    class_room_chosen = ttk.Combobox(window, width=20, height=10, textvariable=class_room, state='readonly')
    class_room_chosen['values'] = room_name
    class_room_chosen.grid(column=0, row=1, sticky='e')

    # 选择课时标签加下拉菜单
    choose_time = tk.Label(window, text="选择课时", width=15, height=2, font=("黑体", 12)).grid(column=0, row=2, sticky='w')
    course_time = tk.StringVar()
    global course_time_chosen
    course_time_chosen = ttk.Combobox(window, width=20, height=10, textvariable=course_time, state='readonly')
    course_time_chosen['values'] = time_name
    course_time_chosen.grid(column=0, row=2, sticky='e')
    var = tk.StringVar()  # tkinter中的字符串
    display = tk.Label(window, textvariable=var, font=('Arial', 12), width=38, height=10)
    display.grid(column=0, row=4, sticky='n')
    #选择是手动还是自动,并调用相应函数

    entry = tk.Entry(window)
    entry.grid(column=0, row=12, sticky='s')

    # # 创建一个函数来处理输入
    # def handle_input():
    #     input_text = entry.get()
    #     print(f"You entered: {input_text}")

    # 创建一个按钮，点击时会调用handle_input函数
    #button = tk.Button(window, text="Submit", command=handle_input)
    #button.pack()
    rate_button = ttk.Button(window, text="分析数据", command=lambda:analyze_data()).grid(column=0, row=4, sticky='s')
    pic_button = ttk.Button(window, text="折线图", command=lambda: draw('./data'+'/'+str(class_room_chosen.get()) + str(course_time_chosen.get())+'/result.xlsx')).grid(column=0, row=5)
    threshold = ttk.Button(window, text="抬头情况", command=lambda: head_up('./data'+'/'+str(class_room_chosen.get()) +str(course_time_chosen.get())+'/result.xlsx')).grid(column=0, row=8)
    threshold = ttk.Button(window, text="到课情况", command=lambda: arrive_rate('./data' + '/' + str(class_room_chosen.get()) +str(course_time_chosen.get()) + '/result.xlsx')).grid(column=0, row=6)
    show_video = ttk.Button(window, text="课堂视频(q键退出,下方可选择从第几分钟开始,默认为0)", command=lambda:show_my_video('./data'+'/'+str(class_room_chosen.get()) +str(course_time_chosen.get())+'/class_video.mp4',entry.get())).grid(column=0, row=11, sticky='s')
    button1 = tk.Button(window, text="展示该时刻上课内容", command=lambda:show_high_class_content('./data'+'/'+str(class_room_chosen.get()) + str(course_time_chosen.get()),'./data'+'/'+str(class_room_chosen.get()) + str(course_time_chosen.get())+'/content_video.mp4')).grid(column=1, row=9, sticky='w')
    button2 = tk.Button(window, text="展示该时刻上课内容", command=lambda:show_low_class_content('./data'+'/'+str(class_room_chosen.get()) + str(course_time_chosen.get()),'./data'+'/'+str(class_room_chosen.get()) + str(course_time_chosen.get())+'/content_video.mp4')).grid(column=1, row=10, sticky='w')

    head_1 = tk.Entry(window)
    head_1.configure(width=30)  # 设置宽度为20
    head_1.grid(column=0, row=9, sticky='s')
    head_2 = tk.Entry(window)
    head_2.configure(width=30)  # 设置宽度为20
    head_2.grid(column=0, row=10, sticky='s')
    arrive = tk.Entry(window)
    arrive.configure(width=30)  # 设置宽度为20
    arrive.grid(column=0, row=7, sticky='s')
    #到课率函数
    def arrive_rate(file):
        # 读取 Excel 文件
        df_1 = pd.read_excel(file)
        human_sum_values = df_1['person_num'].iloc[-1]
        df_2 =pd.read_excel('py_excel.xls')
        filtered_rows = df_2[(df_2['教室'] ==str(class_room_chosen.get()) ) & (df_2['上课时间'] ==str(course_time_chosen.get()))]
        name=human_sum_values/filtered_rows['应到人数'].values
        name  = np.round(name, 2)
        arrive.delete(0, tk.END)
        arrive.insert(0, f"到课率为{name}")

    # 抬头率函数
    def head_up(file):
        # 读取 Excel 文件
        df = pd.read_excel(file)

        # 找出 "value" 列中大于0.5的值对应的 "name" 列中的值
        # names_1 = df[df['value'] > 0.15]['time'].tolist()
        # names_2 = df[df['value'] < 0.1]['time'].tolist()
        max_value = df['value'].max()
        min_value = df['value'].min()

        names_1 = df[df['value'] > max_value * 0.8]['time'].tolist()
        names_2 = df[df['value'] < min_value * 1.2]['time'].tolist()
        #print(names_2)
        # 将这些名字连接成一个字符串，名字之间用逗号分隔
        #names_1 = [str(names_1) for name in names_1]
        #names_2 = [str(names_2) for name in names_2]
        names_1 = [str(name) for name in names_1]
        names_2 = [str(name) for name in names_2]
        #print(names_2)
        result_1 = ', '.join(names_1)
        result_2 = ', '.join(names_2)
        head_1.delete(0, tk.END)
        head_1.insert(0, f"抬头率过低的时间‘{result_1}’")
        head_2.delete(0, tk.END)
        head_2.insert(0, f"抬头率高的时间‘{result_2}’")
        directory = os.path.dirname(file)
        filename_1 = os.path.join(directory, 'high.txt')
        filename_2 = os.path.join(directory, 'low.txt')

    # 写入结果到文本文件
        with open(filename_1, 'w') as file_1:
            file_1.write(result_1)

        with open(filename_2, 'w') as file_2:
            file_2.write(result_2)
        #print(result_1)
        #print(result_2)
    # Adding a Button
    #rate_button = ttk.Button(window, text="Get_rate", command=rate_cal).grid(column=0, row=4, sticky='s')

    #pic_button = ttk.Button(window, text="折线图", command=draw('output_human.xlsx')).grid(column=0, row=5)
    window.mainloop()

    
# 连接数据库
conn = pymysql.connect(
    host='bj-cdb-qezs3dji.sql.tencentcdb.com',
    port=63842,
    user='root',
    password='openCV!1',
    database='opencv',
    charset='utf8'
)
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

# 创建主窗口
window = tk.Tk()
window.title('欢迎来到注册登录界面')
window.geometry('400x300')

# 创建登录界面的Frame
login_frame = tk.Frame(window)
login_frame.place(x=0, y=0, width=400, height=300)

# 创建登录界面的小部件
tk.Label(login_frame, text='账号：').grid(row=0, column=0, padx=10, pady=10)
tk.Label(login_frame, text='密码：').grid(row=1, column=0, padx=10, pady=10)
login_usr = tk.StringVar()
login_pwd = tk.StringVar()
tk.Entry(login_frame, textvariable=login_usr).grid(row=0, column=1, padx=10, pady=10)
tk.Entry(login_frame, textvariable=login_pwd, show='*').grid(row=1, column=1, padx=10, pady=10)
tk.Button(login_frame, text='登录', command=lambda: login()).grid(row=2, column=0, padx=10, pady=10)
tk.Button(login_frame, text='注册', command=lambda: switch_to_register()).grid(row=2, column=1, padx=10, pady=10)

# 创建注册界面的Frame
register_frame = tk.Frame(window)
register_frame.place(x=0, y=0, width=400, height=300)

# 创建注册界面的小部件
tk.Label(register_frame, text='用户名：').grid(row=0, column=0, padx=10, pady=10)
tk.Label(register_frame, text='密码：').grid(row=1, column=0, padx=10, pady=10)
tk.Label(register_frame, text='确认密码：').grid(row=2, column=0, padx=10, pady=10)
reg_usr = tk.StringVar()
reg_pwd1 = tk.StringVar()
reg_pwd2 = tk.StringVar()
tk.Entry(register_frame, textvariable=reg_usr).grid(row=0, column=1, padx=10, pady=10)
tk.Entry(register_frame,textvariable = reg_pwd1 ,show ='*').grid(row = 1 ,column = 1,padx = 10,pady = 10)
tk.Entry(register_frame,textvariable = reg_pwd2 ,show ='*').grid(row = 2 ,column = 1,padx = 10,pady = 10)
tk.Button(register_frame,text ='注册',command=lambda: register()).grid(row = 3,column = 0,padx = 10,pady = 10)
tk.Button(register_frame,text ='返回',command=lambda: switch_to_login()).grid(row = 3,column = 1,padx = 10,pady = 10)

# 隐藏注册界面
register_frame.place_forget()

# 定义登录函数
def login():
    # 获取用户输入的账号密码
    log_usr = login_usr.get()
    log_pwd = login_pwd.get()
    # 判断账号密码是否为空
    if not (log_usr and log_pwd):
        messagebox.showerror('错误', '账号密码不能为空')
        return
    # 执行SQL语句，查询用户是否存在
    sql = "select * from user where usr=%s and pwd=%s;"
    if cursor.execute(sql,(log_usr ,log_pwd)):
        messagebox.showinfo('成功', '登录成功!')
        # 关闭窗口
        window.destroy()
        get_in()
    else:
        messagebox.showerror('错误', '账号或密码错误!')

# 定义注册函数
def register():
    # 获取用户输入的用户名和密码
    usr = reg_usr.get()
    pwd1 = reg_pwd1.get()
    pwd2 = reg_pwd2.get()
    # 判断用户名和密码是否为空
    if not (usr and pwd1 and pwd2):
        messagebox.showerror('错误', '输入不能为空')
        return
    # 判断两次密码是否一致
    if not pwd1 == pwd2:
        messagebox.showerror('错误', '两次密码不一致')
        return
    # 执行SQL语
    sql ='insert into user(usr, pwd) values(%s,%s)'
    cursor.execute(sql,(usr,pwd1))
    conn.commit()  # 提交数据到数据库并保存
    messagebox.showinfo('成功', '注册成功') # 切换到登录界面
    switch_to_login()

def switch_to_register():
    login_frame.place_forget() # 显示注册界面
    register_frame.place(x=0, y=0, width=400, height=300)

def switch_to_login(): # 隐藏注册界面
    register_frame.place_forget() # 显示登录界面
    login_frame.place(x=0, y=0, width=400, height=300)

window.mainloop()