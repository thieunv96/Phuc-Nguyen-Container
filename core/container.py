import cv2
import numpy as np 
import glob



def processing(img, name = "NoName.jpg", debug = False):
    h , w, _ = img.shape
    # k = int(min(w, h) / 10)
    # k = k if k % 2 != 0 else k + 1
    img_roi = img[ int(h/10):int(h/2), int(w/2):w]
    img_pros = cv2.bilateralFilter(img_roi, 9, 49, 49)
    kernel = np.ones((5,5),np.uint8)
    img_pros = cv2.erode(img_pros, kernel)
    img_gray = cv2.cvtColor(img_pros, cv2.COLOR_BGR2GRAY)
    sobelx64f = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=1)
    abs_sobel64f = np.absolute(sobelx64f)
    img_sobel = np.uint8(abs_sobel64f)
    _, img_threshold = cv2.threshold(img_sobel, 40, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        xb, yb, wb, hb = cv2.boundingRect(cnt)
        s = cv2.contourArea(cnt)
        if(hb > 100 or s < 10):
            cv2.drawContours(img_threshold, [cnt], 0, (0), -1)

    kernel_close = np.ones((1,107),np.uint64)
    img_closed = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((11,11),np.uint64)
    img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_open)
    
    _, contours, _ = cv2.findContours(img_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    width_box = lambda cnt: cv2.boundingRect(cnt)[2]
    contours = sorted(contours, key=width_box, reverse=True)
    # xb, yb, wb, hb = cv2.boundingRect(contours[0])
    # img_pros = cv2.rectangle(img_roi, (xb, yb), (xb+wb, yb+hb), (0,0,255), 1)

    for i in range(len(contours)):
        _, yi, _, _ = cv2.boundingRect(contours[i])
        _, y0, _, _ = cv2.boundingRect(contours[0])
        if(yi<y0):
            cv2.drawContours(img_opened, contours,i , (0), -1)
    _, contours, _ = cv2.findContours(img_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_box = lambda cnt: cv2.boundingRect(cnt)[1]
    contours = sorted(contours, key=y_box, reverse=False)
    x1, y1, w1, h1 = cv2.boundingRect(contours[0])
    x2, y2, w2, h2 = cv2.boundingRect(contours[1])
    xc, yc, wc, hc = x1 - 10, y1 - 10, w1 + 20, h2 + y2 - y1 + 20
    img_result = cv2.rectangle(img_roi, (xc, yc), (xc+wc, yc+hc), (0,0,255), 1)
    img_crop = img_result[yc:yc+hc, xc:xc+wc]
    kernel = np.ones((3,3),np.uint8)
    img_remove_bound = cv2.morphologyEx(img_crop, cv2.MORPH_OPEN, kernel)
    img_gray_rb = cv2.cvtColor(img_remove_bound, cv2.COLOR_BGR2GRAY)
    _, img_threshold_result = cv2.threshold(img_gray_rb, 170, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((1,k),np.uint64)
    # result = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel)
    # k = int(k/2) if int(k/2) % 2 != 0 else int(k/2) + 1
    # kernel = np.ones((1,k),np.uint8)
    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    # _, contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     xb, yb, wb, hb = cv2.boundingRect(cnt)
    #     if(hb > 30):
    #     # if(wb > w/4):
    #         cv2.rectangle(img_pros, (xb, yb), (xb+wb, yb+hb), (0,255,0), 3)
    # img[int(h/10):int(h/2), int(w/2):w] = img_pros
    # img_blured = cv2.medianBlur(img_pros, 83)
    # img_sub = cv2.absdiff(img_pros, img_blured)
    # kernel = np.ones((k,k),np.uint8)
    # img_dilated = cv2.dilate(img_pros, kernel)
    # img_eroded = cv2.erode(img_pros, kernel)
    # img_abs = cv2.absdiff(img_dilated, img_eroded)
    # img_gray = cv2.cvtColor(img_abs, cv2.COLOR_BGR2GRAY)
    # _, img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    # # img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel)
    # _, contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     xb, yb, wb, hb = cv2.boundingRect(cnt)
    #     if(hb > 300):
    #         cv2.drawContours(img_threshold, [cnt], 0, (0), -1)
    cv2.imwrite("./out/" + name, img_threshold_result)


def test():
    files = glob.glob("../images/*.jpg")
    for f in files:
        img = cv2.imread(f, 1)
        name = f.split("\\")[-1]
        processing(img, name = name)

if __name__ == "__main__":
    test()
