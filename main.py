from tkinter import *
import tkinter as tk
from tkinter import ttk
import tkinter
from tkinter import messagebox
import threading
import cv2
import numpy as np
import torch
from tkinter.filedialog import *
import tkinter.colorchooser
from PIL import Image, ImageTk
from model import U2NET
import winreg

def get_desktop():
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                          r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders',)
    return winreg.QueryValueEx(key, "Desktop")[0]

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def get_img(pic_path, width, height):
    pic = Image.open(pic_path).resize((width, height))
    pic = ImageTk.PhotoImage(pic)
    return pic


class Operate(object):
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title(u"OutBreak-Hui&证件照编辑")
        self.root.geometry("600x380")
        self.root.resizable(width=False, height=False)

        self.picDir = ""
        self.color = None
        self.save_height = None
        self.save_width = None
        self.save_img = None

        self.lab1 = Label(self.root, text=u"选择图片")
        self.entry1 = Entry(self.root, width=60)
        self.entry1.delete(0, "end")
        self.B1 = Button(self.root, text=u"浏览", command=self.show_pic, width=7)

        self.lab2 = Label(self.root, text=u"选择尺寸", width=43)
        self.var1 = tkinter.StringVar()
        self.combobox = tkinter.ttk.Combobox(self.root, textvariable=self.var1, value=(u"一寸", u"小一寸", u"大一寸",
                                                                                       u"二寸", u"五寸"), width=41)
        self.B2 = Button(self.root, text=u"选择背景", command=self.ChooseColor, width=43)
        self.B3 = Button(self.root, text=u"转   换&预   览", command=self.change_background, width=43)
        self.B4 = Button(self.root, text=u"保   存", command=self.save_pic, width=43)

        self.canva_show = Canvas(self.root, width=240, height=320, bg="white")

        global img
        img = get_img("select.png", 300, 100)
        self.lab3 = Label(self.root, image=img)

    def gui_arrang(self):
        self.lab1.place(x=20, y=10, anchor='nw')
        self.B1.place(x=529, y=7, anchor='nw')
        self.entry1.place(x=90, y=10, anchor='nw')

        self.canva_show.place(x=20, y=40, anchor='nw')

        self.B2.place(x=275, y=41, anchor='nw')

        self.lab2.place(x=275, y=81, anchor='nw')
        self.combobox.place(x=275, y=121, anchor='nw')

        self.B3.place(x=275, y=161, anchor='nw')

        self.B4.place(x=275, y=201, anchor='nw')

        self.lab3.place(x=275, y=251, anchor="nw")

    def get_pic_dir(self):
        default_dir = r"文件路径"
        self.picDir = askopenfilename(title=u"选择文件", initialdir=(os.path.expanduser(default_dir)))
        self.entry1.insert(0, str(self.picDir))
        return self.picDir

    def show_pic(self):
        global img_tk
        picDir = self.get_pic_dir()
        # img = cv2.imread(picDir)
        img = cv2.imdecode(np.fromfile(picDir, dtype=np.uint8), -1)
        # print(img)
        canvawidth = int(self.canva_show.winfo_reqwidth())
        canvaheight = int(self.canva_show.winfo_reqheight())

        img = cv2.resize(img, (canvawidth, canvaheight), interpolation=cv2.INTER_AREA)
        imgcv2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        current_img = Image.fromarray(imgcv2)
        img_tk = ImageTk.PhotoImage(image=current_img)
        self.canva_show.create_image(0, 0, anchor='nw', image=img_tk)

    def ChooseColor(self):
        r = tkinter.colorchooser.askcolor(title="颜色选择器")
        r = r[0]
        self.color = r
        # return r

    def thread_creat(self, func, *args):
        t = threading.Thread(target=func, args=args)
        t.setDaemon(True)
        t.start()

    def change_background(self):
        global img_tk
        pic_path = self.picDir
        model_name = "u2net"
        model_dir = "saved_models\\u2net\\u2net.pth"

        net = U2NET(3, 1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location="cpu"))

        net.eval()

        pic = cv2.imdecode(np.fromfile(pic_path, dtype=np.uint8), -1)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic_file = pic
        size = (pic.shape[1], pic.shape[0])

        pic = cv2.resize(pic, (320, 320)).astype(np.float32)
        pic /= 255.0
        pic = ((pic - mean) / std).astype(np.float32)
        pic = torch.from_numpy(pic.transpose(2, 0, 1)).unsqueeze(0)


        if torch.cuda.is_available():
            pic = pic.cuda()
        else:
            pass

        d1, d2, d3, d4, d5, d6, d7 = net(pic)
        pred = d1[:, 0, :, :]

        pred = normPRED(pred)

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        predict_np = cv2.erode(predict_np, kernel)
        predict_np = cv2.GaussianBlur(predict_np, (3, 3), 0)

        im = Image.fromarray(predict_np * 255).convert('RGB')
        im = im.resize(size, resample=Image.BILINEAR)
        im = np.array(im)
        res = np.concatenate((pic_file, im[:, :, [0]]), -1)
        img = Image.fromarray(res.astype('uint8'), mode='RGBA')

        background = self.color
        color_list = list(background)
        color_list = [int(i) for i in color_list]
        background = tuple(color_list)

        base_image = Image.new("RGB", size, background)

        scope_map = np.array(img)[:, :, -1] / 255
        scope_map = scope_map[:, :, np.newaxis]
        scope_map = np.repeat(scope_map, repeats=3, axis=2)
        res_image = np.multiply(scope_map, np.array(img)[:, :, :3]) + np.multiply((1 - scope_map),
                                                                                  np.array(base_image))
        canvawidth = int(self.canva_show.winfo_reqwidth())
        canvaheight = int(self.canva_show.winfo_reqheight())
        self.save_img = Image.fromarray(np.uint8(res_image))
        res_image = cv2.resize(res_image, (canvawidth, canvaheight), interpolation=cv2.INTER_AREA)
        current_img = Image.fromarray(np.uint8(res_image))
        img_tk = ImageTk.PhotoImage(image=current_img)
        self.canva_show.create_image(0, 0, anchor='nw', image=img_tk)

    def save_pic(self):
        if self.picDir == "":
            tkinter.messagebox.showinfo("提示", "请选择图片")
        if self.combobox.get() == "一寸":
            self.save_height = 413
            self.save_width = 295
            self.save_img = self.save_img.resize((self.save_width, self.save_height), resample=Image.BILINEAR)
            out_path = os.sep.join([get_desktop(), "certificate.jpg"])
            self.save_img.save(out_path, quality=95, subsampling=0)
        elif self.combobox.get() == "小一寸":
            self.save_height = 390
            self.save_width = 260
            self.save_img = self.save_img.resize((self.save_width, self.save_height), resample=Image.BILINEAR)
            out_path = os.sep.join([get_desktop(), "certificate.jpg"])
            self.save_img.save(out_path, quality=95, subsampling=0)
        elif self.combobox.get() == "大一寸":
            self.save_height = 567
            self.save_width = 390
            self.save_img = self.save_img.resize((self.save_width, self.save_height), resample=Image.BILINEAR)
            out_path = os.sep.join([get_desktop(), "certificate.jpg"])
            self.save_img.save(out_path, quality=95, subsampling=0)
        elif self.combobox.get() == "二寸":
            self.save_height = 636
            self.save_width = 413
            self.save_img = self.save_img.resize((self.save_width, self.save_height), resample=Image.BILINEAR)
            out_path = os.sep.join([get_desktop(), "certificate.jpg"])
            self.save_img.save(out_path, quality=95, subsampling=0)
        elif self.combobox.get() == "五寸":
            self.save_height = 1200
            self.save_width = 840
            self.save_img = self.save_img.resize((self.save_width, self.save_height), resample=Image.BILINEAR)
            out_path = os.sep.join([get_desktop(), "certificate.jpg"])
            self.save_img.save(out_path, quality=95, subsampling=0)
        else:
            tkinter.messagebox.showinfo("提示", "请选择转换尺寸")


if __name__ == '__main__':
    O = Operate()
    O.gui_arrang()
    tkinter.mainloop()





