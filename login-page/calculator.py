import tkinter as tk
from tkinter import ttk
import math
from PIL import Image, ImageTk, ImageDraw
import time

class AnimatedButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.flower_img = None
        self.animation_id = None
        self.bind('<Button-1>', self.start_animation)

    def start_animation(self, event):
        if self.animation_id:
            self.after_cancel(self.animation_id)
        
        # 创建花朵图像
        flower_size = 30
        img = Image.new('RGBA', (flower_size, flower_size), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        
        # 绘制花朵动画帧
        for i in range(5):
            angle = i * 72
            x = flower_size//2 + int(10 * math.cos(math.radians(angle)))
            y = flower_size//2 + int(10 * math.sin(math.radians(angle)))
            draw.ellipse((x-3, y-3, x+3, y+3), fill='#FF69B4')
        
        self.flower_img = ImageTk.PhotoImage(img)
        self.flower_label = tk.Label(self, image=self.flower_img, bg=self['bg'])
        self.flower_label.place(x=event.x-15, y=event.y-15)
        
        # 0.5秒后移除花朵
        self.animation_id = self.after(500, self.remove_flower)

    def remove_flower(self):
        self.flower_label.destroy()
        self.animation_id = None

class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("花園計算機")
        self.root.geometry("350x500")
        self.root.configure(bg='#98FB98')  # 粉綠色背景
        
        # 背景 - 草地
        self.canvas = tk.Canvas(root, bg='#98FB98', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 顯示區
        self.display_var = tk.StringVar()
        display_frame = tk.Frame(self.canvas, bg='#FFB6C1', bd=5, relief=tk.SUNKEN)
        display_frame.place(relx=0.5, rely=0.1, relwidth=0.8, relheight=0.15, anchor='n')
        
        display = tk.Entry(display_frame, textvariable=self.display_var, 
                         font=('Arial', 20), justify='right', bd=0, bg='#FFE4E1',
                         fg='#FF0000')  # 紅色文字
        display.pack(fill=tk.BOTH, expand=True)
        
        # 按鈕區
        buttons = [
            ('7', 0.2, 0.3), ('8', 0.4, 0.3), ('9', 0.6, 0.3), ('+', 0.8, 0.3), ('√', 0.9, 0.3),
            ('4', 0.2, 0.45), ('5', 0.4, 0.45), ('6', 0.6, 0.45), ('-', 0.8, 0.45), ('x²', 0.9, 0.45),
            ('1', 0.2, 0.6), ('2', 0.4, 0.6), ('3', 0.6, 0.6), ('×', 0.8, 0.6), ('C', 0.9, 0.6),
            ('0', 0.2, 0.75), ('.', 0.4, 0.75), ('=', 0.6, 0.75), ('÷', 0.8, 0.75)
        ]
        
        for (text, x, y) in buttons:
            btn = AnimatedButton(self.canvas, text=text, font=('Arial', 15), 
                               bg='#90EE90', activebackground='#98FB98',
                               command=lambda t=text: self.on_button_click(t))
            btn.place(relx=x, rely=y, relwidth=0.15, relheight=0.12, anchor='n')  # 增加按鈕高度避免重疊
        
        # 添加花朵裝飾
        self.add_decorations()
        
    def add_decorations(self):
        # 在背景添加隨機花朵
        for _ in range(10):
            x = 50 + 250 * (_ % 5) / 5
            y = 400 + 50 * (_ % 2)
            self.canvas.create_oval(x, y, x+10, y+10, fill='#FF69B4', outline='')
    
    def on_button_click(self, char):
        current = self.display_var.get()
        
        if char == 'C':
            self.display_var.set('')
        elif char == '=':
            try:
                result = eval(current.replace('×', '*').replace('÷', '/'))
                self.display_var.set(str(result))
            except:
                self.display_var.set('Error')
        elif char == 'x²':
            try:
                result = float(current) ** 2
                self.display_var.set(str(result))
            except:
                self.display_var.set('Error')
        elif char == '√':
            try:
                result = math.sqrt(float(current))
                self.display_var.set(str(result))
            except:
                self.display_var.set('Error')
        else:
            self.display_var.set(current + char)

if __name__ == "__main__":
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()
