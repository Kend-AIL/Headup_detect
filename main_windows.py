import pymysql
import tkinter as tk
from tkinter import messagebox

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