from tkinter import *
from PIL import Image, ImageTk
import os, os.path
import random
import time

from tkinter import filedialog


img_list = []

class Window(Frame):
    def __init__(self, master=None, path=''):
        Frame.__init__(self, master)
        self.path = path
        self.master = master
        self.pack(fill=BOTH, expand=1)
        self.img_name = self.get_img_name(self.path)
        self.image_path = path + self.img_name
        self.img = Label(image='')
        self.pred_caption = Label(text='')
        self.setimage()
        self.predictbutton()
        self.button()
        
    
    def get_img_name(self, path):
        global img_list
        img_names = os.listdir(self.path)
        img_name = random.choice(img_names)
        if len(img_list) == len(img_names):
            img_list = []
        while img_name in img_list:
            img_name = random.choice(img_names)
        img_list.append(img_name)
        # print(img_list)
        return img_name
    
    def img_resize(self, img):
        width = img.size[0]
        height = img.size[1]
        boundary = 500
        
        # if  width < height:
        #     resized_image = img.resize((int(float(width) * float(boundary/height)),boundary),
        #                                      Image.ANTIALIAS)
        # else:
        #     resized_image = img.resize((boundary,int(float(height) * float(boundary/width))),
        #                                      Image.ANTIALIAS)
        resized_image = img.resize((int(1.49*300), 300),Image.ANTIALIAS)
        return resized_image
        
    def button(self):
        btn = Button(self, text="Next Image", command=self.on_click)
        btn.place(x=200,y=500)
        
    def predictbutton(self):
        btn = Button(self, text="Predict", command=self.predict)
        btn.place(x=100,y=500)
      
    def predict(self):
        os.system(f'python sample.py --image={self.image_path}')

        time.sleep(10)
        result_image = "./results/predicted.png"
        self.image_path = result_image
        self.setimage(show=True)



    def on_click(self):
        self.update_idletasks()
        self.img_name = self.get_img_name(self.path)
        self.image_path = path + self.img_name
        self.img.config(image='')
        self.pred_caption.config(text='')
        self.setimage(show=False)
        
        #self.setlabel(display=False)
                      
    def setimage(self, show=True):
        load = Image.open(self.image_path)
        load = self.img_resize(load)
        
        
        if show:
            render = ImageTk.PhotoImage(load)
            img = Label(self, image=render)
            img.config(image=render)
            img.image = render

        else:
            render = ImageTk.PhotoImage(load)
            img = Label(self, image=render)
            self.img.config(image=render)
            self.img.image = render
        img.place(x=0, y=0)
     

        
    
    
if __name__ == '__main__':

    root = Tk()
    folder_path = filedialog.askdirectory()
    path = folder_path + '/'
    app = Window(root,path)
    root.wm_title("Food Detection")
    root.geometry("500x5000")
    root.mainloop()
