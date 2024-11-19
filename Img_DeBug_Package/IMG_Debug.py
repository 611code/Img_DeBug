import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
'''
功能:获取指定颜色的hsv值
参数:
    img_BGR:BRG格式的图像
    distribution:分布方式
        1:水平分布
        0:垂直分布
    func:滑动栏绑定函数
    size:列表,内含两个数值(宽,高),图像的尺寸
    part:滑动栏是否与图像分离
        0:不分离
        1:分离
'''
def empty(a):
    pass
def Get_Appoint_Color(img_BGR,distribution=1,func=empty,size=[200,200],part=0):
    if not isinstance(img_BGR,np.ndarray):
        print("这不是有效的图像")
        return -1
    print("点击'q'退出")
    img_BGR = cv2.resize(img_BGR,size)
    imgHSV=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
    cv2.namedWindow("TrackBars")                               # 生成一个叫 name 的窗口
    if not part:
        cv2.resizeWindow("TrackBars",[size[0]*4,size[1]*2])          # 改变其窗口大小
    else:
        cv2.resizeWindow("TrackBars",[640,240]) 
    cv2.createTrackbar("Hue Min","TrackBars",0,179,func)       # 为窗口创建一个0~179的滑动条,0为起始位置
    cv2.createTrackbar("Hue Max","TrackBars",179,179,func)     # 为窗口创建一个0~179的滑动条
    cv2.createTrackbar("Sat Min","TrackBars",0,255,func)      # 为窗口创建一个0~255的滑动条
    cv2.createTrackbar("Sat Max","TrackBars",255,255,func)     # 为窗口创建一个0~255的滑动条
    cv2.createTrackbar("val Min","TrackBars",0,255,func)      # 为窗口创建一个0~255的滑动条
    cv2.createTrackbar("val Max","TrackBars",255,255,func)     # 为窗口创建一个0~255的滑动条
    while True:
        h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("val Max", "TrackBars")

        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask  = cv2.inRange(imgHSV,lower,upper)                # 相当与蒙版，用来与源图片比较
        imgResult = cv2.bitwise_and(img_BGR,img_BGR,mask=mask)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if distribution:
            horizontal_image_1 = np.hstack((img_BGR,imgHSV))
            horizontal_image_2 = np.hstack((mask,imgResult))
            vertical_image = np.hstack((horizontal_image_1, horizontal_image_2))
        else:
            horizontal_image_1 = np.hstack((img_BGR,imgHSV))
            horizontal_image_2 = np.hstack((mask,imgResult))
            vertical_image = np.vstack((horizontal_image_1, horizontal_image_2))
        if part==1:
            cv2.imshow("src-hsv/mask/result",vertical_image)
        else:
            cv2.imshow('TrackBars',vertical_image)
        key = cv2.waitKey(1)                                    # 10表示暂停 10ms ,0表示一直停止
        if key & 0xFF== ord('q'):
            cv2.destroyAllWindows()
            print("已退出")
            break
'''
功能:显示多个图像在一个窗口
参数:
    imgs:列表,包含你要显示的图像序列
    window_name:你要显示窗口的名称
    lines:每行显示的图像数
    scale:图像的比例
'''
def CV2_Imgs(imgs,window_name="imgs",lines=3,scale=1):
    max_height = int(max(image.shape[0] for image in imgs)*scale)
    max_width = int(max(image.shape[1] for image in imgs)*scale)

    resized_images = []
    for image in imgs:
        if len(image.shape) == 2:  # 如果是灰度图像，扩展为三通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resized_image = cv2.resize(image, (max_width, max_height))
        resized_images.append(resized_image)
    num_rows = (len(resized_images) + lines - 1) // lines

    grid_rows = []
    for i in range(num_rows):
        start_idx = i * lines
        end_idx = min(start_idx + lines, len(resized_images))
        row_images = resized_images[start_idx:end_idx]
        if len(row_images) < lines:
            # 如果最后一行图像数量不足，用空白图像补齐
            for _ in range(lines - len(row_images)):
                row_images.append(np.zeros_like(resized_images[0]))
        grid_rows.append(np.hstack(row_images))
    
    combined_image = np.vstack(grid_rows)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
功能:显示多个 cv2 格式的图像在一个图形中。
参数:
    - images: 图像数据的列表，每个图像都是 numpy.ndarray 类型。
    - cols: 每行显示的图像数量，默认为3。
    - figsize: 图形的大小，默认为(15, 10)。
    - titles: 图像标题的列表，如果提供，长度应与图像数量相同。
"""
def PLT_Imgs(images, lines=3, figsize=(15, 10), titles=None):
    # 计算行数
    if lines <=2:
        lines = 2
    num_images = len(images)
    rows = int(num_images / lines) + (num_images % lines > 0)
    # 创建图形和子图
    fig, axs = plt.subplots(rows, lines, figsize=figsize)
    # 如果只有一个子图，将其转换为列表
    if isinstance(axs, plt.Axes):
        axs = [axs]
    # 展平子图列表
    axs = axs.flatten()
    # 显示图像
    for i, img in enumerate(images):
        # 将 BGR 图像转换为 RGB 图像
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img_rgb)
        if titles and i < len(titles):
            axs[i].set_title(titles[i])
        axs[i].axis('off')
    # 隐藏多余的子图
    for j in range(num_images, len(axs)):
        fig.delaxes(axs[j])
    # 显示图形
    plt.show()

def Data_DeBug(msg,data):
    try:
        shape = data.shape
    except:
        shape = None
    file_name = os.path.basename(__file__)
    line_number = sys._getframe().f_lineno
    print(f"[Message:{msg}, data:{data}, shape:{shape}, File:{file_name}, Line:{line_number}]")



if __name__ == '__main__':
    img = np.random.randint(0, 255, size=(30,30,3), dtype=np.uint8)
    # CV2_Imgs([img,img,img,img],window_name="imgs",lines=3,scale=0.9)
    # print(type(img))
    Get_Appoint_Color(img,distribution=0,size=[300,300],part=1)

    # PLT_Imgs([img,img,img,img],lines=5)

    # a = 10
    # Data_DeBug("看看a",a)





