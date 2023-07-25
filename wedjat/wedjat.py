import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

import imutils

'''
Preparation of the image for the detection of the registers.
'''
def prep_image(filename,img_file=True):
    if img_file:
        img_hiero_txt = cv2.imread(filename)
    else:
        img_hiero_txt = np.array(filename)
    
    original_h,original_w = img_hiero_txt.shape[:2]

    # if original_h > 800:
    #     img_hiero_txt = cv2.resize(img_hiero_txt, None,fx=800/original_h,fy = 600/original_w, interpolation= cv2.INTER_LINEAR)
    #     original_h,original_w = img_hiero_txt.shape[:2]

    return img_hiero_txt,original_h,original_w 


def get_signs(img_hiero_txt):
    gray = cv2.cvtColor(img_hiero_txt, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    _x = 1
    _y = 1
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_x, _y))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

    # plt.imshow(dilation)
    # plt.show()   

    cnts, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
    
    # cnts = imutils.grab_contours(contours)

    if cnts == []:
        return [],[]

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b: b[1][0], reverse=False)) 

    return boundingBoxes

    # out_registers = []
    # for cnt in cnts:
    #     x, y, w, h = cv2.boundingRect(cnt)                
    #     out_registers.append([x, y, w, h])

    # return out_registers

'''
The following function is used to get the horizontal registers from an image.
It takes as input the name of the file and returns the image with the bounding boxes
around each register.
'''
def get_horizontal_registers(img_hiero_txt,original_h,original_w, blur=(9,9) ,height_threshold=0.85,_iter=1):
    
    gray = cv2.cvtColor(img_hiero_txt, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    _x = original_w
    _y = 1
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_x, _y))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations = _iter) 

    # plt.imshow(dilation)
    # plt.show()   

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)
    
    # Get the bounding box for each register
    out_registers = []
    for cnt in reversed(contours):
        x, y, w, h = cv2.boundingRect(cnt)                
        out_registers.append([x, y, w, h])
        
    avg_reg_height = np.average(np.array(out_registers)[:,3])

    # remove registers that are smaller than 85% (height_threshold) of the average height (reduce smaller error register)
    out_registers_clean = np.array(out_registers)[np.array(out_registers)[:,3] > avg_reg_height*height_threshold]

    print("| Number of registers identified: ",len(out_registers))
    print("| Number of clean registers:",len(out_registers_clean))

    return out_registers_clean.tolist()

'''
Processes an horizontal register into vertical clusters of signs and sorts these top to bottom
'''
def get_sign_cluster(reg_img,h):
   # Get horizontal clusters of signs
    gray = cv2.cvtColor(reg_img, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    _x = 1
    _y = h
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_x, _y))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

    # plt.imshow(dilation)
    # plt.show()   

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)       
    
    signs_out = []
    hiero_same_reference = []
    for cnt in reversed(contours):
        x0, y0, w0, h0 = cv2.boundingRect(cnt)

        # print("Sign cluster coordinates: (%d,%d,%d,%d) on the register coordinates"%(x0,y0,w0,h0))

        crop_cluster_img = reg_img[y0:y0+h0, x0:x0+w0]

        # plt.imshow(reg_img)
        # plt.show()   
        # plt.imshow(crop_cluster_img)
        # plt.show()           

        crop_cluster_img = cv2.copyMakeBorder(crop_cluster_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT,value=[255,255,255])

        crop_cluster_img,or_h,or_w  = prep_image(crop_cluster_img,False)

        # plt.imshow(crop_cluster_img)
        # plt.show()          

        # Get the hieroglyphs in the register
        signs = get_signs(crop_cluster_img)

        signs = sorted(signs, key=lambda b: b[1], reverse=False)
        # print(sorted(signs, key=lambda b: b[1], reverse=False))

        bb_on_original_ref = np.array(signs)+np.array([x0-1,y0-1,0,0])
        # bb = np.array(signs)

        # print("--->",np.array(signs))

        # print(bb_on_original_ref)

        signs_out.extend(bb_on_original_ref)
        # hiero_same_reference.extend(bb)

    # print("signs_out",signs_out)
    # print("signs_out2",signs_out.tolist())      

        # break 
    return signs_out


'''
Overlay registers on the image.
'''
def print_registers(reg_arr,img_hiero,signs="",print=True):
    # Draw a green bounding box around each register and print out with imshow
    im2 = img_hiero.copy()
    # print(reg_arr)
    for idx,bb in enumerate(reg_arr):
        x, y, w, h = bb

        # Drawing a rectangle on copied image
        im = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if signs != "":
            # Drawing sign on
            cv2.putText(
                im2,
                signs[idx],
                (x+2, y+10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )
    if print:
        plt.figure(figsize=(50,50))
        plt.imshow(im)
        plt.show()

    return im

def print_registers_with_sign(signs,img_hiero):
    # Draw a green bounding box around each register and print out with imshow
    im2 = img_hiero.copy()

    for idx,bb in enumerate(signs):
        x, y, w, h = bb[1]
        sign = bb[0]

        text = str(sign)

        # Drawing a rectangle on copied image
        im = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
   
        # Drawing sign on
        cv2.putText(
            im2,
            sign,
            (x+2, y+10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )


    plt.figure(figsize=(100,100))
    plt.imshow(im)
    plt.show()

    return im 

