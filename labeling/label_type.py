from tkinter import *
from tkinter.colorchooser import askcolor


class Paint(object):
    DEFAULT_PEN_SIZE = 50
    DEFAULT_COLOR = 'black'

    def __init__(s):
        s.root = Tk()

        s.pen_button = Button(s.root, text='pen', command=s.use_pen)
        s.pen_button.grid(row=0, column=0)

        s.brush_button = Button(s.root, text='brush', command=s.use_brush)
        s.brush_button.grid(row=0, column=1)

        s.color_button = Button(s.root, text='color', command=s.choose_color)
        s.color_button.grid(row=0, column=2)

        s.eraser_button = Button(s.root, text='eraser', command=s.use_eraser)
        s.eraser_button.grid(row=0, column=3)

        s.choose_size_button = Scale(s.root, from_=1, to=100, orient=HORIZONTAL)
        s.choose_size_button.grid(row=0, column=4)

        s.c = Canvas(s.root, bg='white', width=600, height=600)
        s.c.grid(row=1, columnspan=5)

        s.setup()
        s.root.mainloop()

    def setup(s):
        s.old_x = None
        s.old_y = None
        s.line_width = s.choose_size_button.get()
        s.color = s.DEFAULT_COLOR
        s.eraser_on = False
        s.active_button = s.pen_button
        s.c.bind('<B1-Motion>', s.paint)
        s.c.bind('<ButtonRelease-1>', s.reset)

    def use_pen(s):
        s.activate_button(s.pen_button)

    def use_brush(s):
        s.activate_button(s.brush_button)

    def choose_color(s):
        s.eraser_on = False
        s.color = askcolor(color=s.color)[1]

    def use_eraser(s):
        s.activate_button(s.eraser_button, eraser_mode=True)

    def activate_button(s, some_button, eraser_mode=False):
        s.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        s.active_button = some_button
        s.eraser_on = eraser_mode

    def paint(s, e):
        s.line_width = s.choose_size_button.get()
        paint_color = 'white' if s.eraser_on else s.color
        if s.old_x and s.old_y:
            s.c.create_line(s.old_x, s.old_y, e.x, e.y, width=s.line_width,
                            fill=paint_color, capstyle=ROUND, smooth=TRUE, splinesteps=36)
        s.old_x = e.x
        s.old_y = e.y

    def reset(s, e):
        s.old_x, s.old_y = None, None


if __name__ == '__main__':
    Paint()

# def draw_bbox(e, STATE, bbox_list):
#     global bboxId
#
#     if STATE['click'] == 0:
#         STATE['x'], STATE['y'] = e.x, e.y
#     else:
#         x1, x2 = round(min(STATE['x'], e.x)/SIZE), round(max(STATE['x'], e.x)/SIZE)
#         y1, y2 = round(min(STATE['y'], e.y)/SIZE), round(max(STATE['y'], e.y)/SIZE)
#         for i, j in enumerate(bbox_list):
#             if j == [0, 0, 0, 0]:
#                 bbox_list.pop(i)
#         bbox_list.append([x1, y1, x2, y2])
#     STATE['click'] = 1-STATE['click']
#
#     return

