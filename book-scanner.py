#!/usr/bin/env python3
"""
book-scanner.py
@author: Will Rhodes

"""
import argparse
from pathlib import Path
from fpdf import FPDF
import pytesseract
import cv2
import numpy as np
import re,shutil,imutils,os,time



def start():
    parser = argparse.ArgumentParser(description='Take a directory of text/book photos and convert to a pdf')

    parser.add_argument('-p','--pattern', type=str, help='file pattern', dest='pattern', default="*.jpg")

    parser.add_argument('-d','--directory', required=True, type=str, help='the directory to use', dest='directory')

    parser.add_argument('-i','--images-only', action='store_true', help='just saves the images', dest='images_only')

    parser.add_argument('-b','--bw', action='store_true', help='converts to black and white', dest='decolor')

    args = parser.parse_args()

    args.filename = os.path.basename(args.directory)

    paths = sorted(Path(args.directory).glob(args.pattern))

    tempfolder = Path(args.directory).joinpath(f'temp_{args.filename}_{time.time()}')
    os.mkdir(tempfolder.resolve()) 

    #itterate through paths
    imgs = {}
    for path in paths:
        print(f'\nCleaning {path}')
        img = cv2.imread(str(path.resolve()))
        if args.decolor:
            img = decolor(img)
        img = resize(img,1200)
        img = rotate(img)
        img = confirmContrast(img)
        img = crop(img)
        img_path = saveImg(img,tempfolder,path)
        imgs[img_path] = img
    
    if not args.images_only:
        convert_pdf(args,imgs)
        shutil.rmtree(tempfolder.resolve())


def decolor(img):
    #removes color but allows the img to have rgb elements going forward
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def resize(img,resolution):
    #resizes it to something better
    width = int(img.shape[1])
    height = int(img.shape[0])
    scale = resolution/max(width,height)
    dim = (int(width*scale), int(height*scale))
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 


def addText(preview,text,color,thickness):
    #adds text to img
    size=3
    chr_size = size*10
    ih,iw,_ = preview.shape
    lines = text.split('\n')
    displacement = max([len(x) for x in lines])
    displacement = min(displacement*chr_size/2,iw/2)
    position = [int(iw/2-displacement),int(ih*0.33)]
    
    for line in lines:
        position[1] = position[1] + (2*chr_size)
        preview = cv2.putText(preview, line, tuple(position), cv2.FONT_HERSHEY_PLAIN,size,color,thickness)
    return preview


def display_preview(name,img,text=None,text2=None,copy=True):
    #copies the image and makes it viewable, returns the key pressed as chr
    preview = img
    if copy:
        preview = img.copy()
    
    if text is not None:
        preview = addText(preview,text,(255,0,0),3)
    if text2 is not None:
        preview = addText(preview,text2,(0,0,255),2)

    preview = resize(preview,600)
    cv2.imshow(name,preview)
    return chr(cv2.waitKey())


def brightness_contrast(img, brightness = 0, contrast = 0):
    #adjust image
    buf = img.copy()
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(buf, alpha_b, buf, 0, gamma_b)        

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def confirmContrast(img):
    #ask the user if the contrast is ok
    print('\tConfirm Contrast')
    keys = "\n  w  \na s d\n\nETR\nESC"
    instructions = "{1}^ brightness\n\nv contrast{0}^ contrast\n{1}v brightness\n{2}continue\n{2}exit".format(" "*14," "*12," "*20)

    choice = None
    adjustments = [40,50]
    adjustContrast = {
        'w':lambda adj: [adj[0]+10,adj[1]],
        's':lambda adj: [adj[0]-10,adj[1]],
        'a':lambda adj: [adj[0],adj[1]-10],
        'd':lambda adj: [adj[0],adj[1]+10]
    }

    while (True):
        choice = display_preview('confirmContrast',brightness_contrast(img,brightness=adjustments[0],contrast=adjustments[1]),copy=False,text=keys,text2=instructions)
        #print('\t\tSelected ord={}, chr={}.'.format(ord(choice),choice))
        if choice == chr(13): break
        elif choice == chr(27): exit()
        elif choice in adjustContrast:
            adjustments = adjustContrast[choice](adjustments)
            print('\t\tAdjusting with {}...'.format(adjustments))

    cv2.destroyAllWindows()
    return brightness_contrast(img,brightness=adjustments[0],contrast=adjustments[1])


def autoRotate(img):
    #try to rotate
        try:
            #use google tesseract osd to find the text orientation
            newdata = pytesseract.image_to_osd(img, nice=0)

            rotation = int(re.search('(?<=Rotate: )\d+', newdata).group(0))
            confidence = float(re.search('(?<=Orientation confidence: )\d+\.\d+', newdata).group(0))
        
            if rotation != 0 and confidence <= 0.33:
                print('applying rotation',rotation,confidence)
                img = imutils.rotate_bound(img, rotation)
        finally:
            return img


def confirmRotate(img):
    #ask the user if the rotation are ok
    print('\tConfirm Rotate')
    keys = "R\nETR\nESC"
    instructions = "{0}rotate\n{0}continue\n{0}exit".format(" "*20)

    choice = None
    while True:
        choice = display_preview('confirmRotate',img,text=keys,text2=instructions)
        #print(f'\t\tSelected ord={ord(choice)}, chr={choice}.')
        if choice == chr(13): break
        elif choice == chr(27): exit()
        elif choice == 'r':
            img = imutils.rotate_bound(img, 90)
            print('\t\tRotating...')
    
    cv2.destroyAllWindows()
    return img


def rotate(img):
    #tackle rotation
    img = autoRotate(img)
    img = confirmRotate(img)
    return img


def autoBounds(img):  
    #crops to the contents minus shadows or fingers around the edge

    # Read image and search for contours. 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,threshold = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
    ih, iw = img.shape
    
    #include contours of medium size and not on the edges
    filter = lambda x, y, w, h: x != 0 and y != 0 and x+w < iw and y+h <ih and w < 0.25*iw and h < 0.25*ih and w > 5 and h > 5
    contours = [c for c in contours if filter(*cv2.boundingRect(c))]
    #print('Found {} contours!'.format(len(contours)))

    #init at quarters around the image
    x1 = iw*0.25
    y1 = ih*0.25
    x2 = iw*0.75
    y2 = ih*0.75
    padding = 5

    #find the maximas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if(x < x1):
            x1 = x-padding
        if(y<y1):
            y1 = y-padding
        if(x+w > x2):
            x2 = x+w+padding
        if(y+h > y2):
            y2 = y+h+padding

    #take care of bounds
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > iw:
        x2 = iw
    if y2 > ih:
        y2 = ih

    return [int(i) for i in [y1,y2,x1,x2]]


def confirmBounds(img,y1,y2,x1,x2):
    #ask the user if the bounds are ok
    print('\tConfirm Bounds')
    choice = None
    for cardinal in range(4):
        print(f'\t\tAdjusting {cardinal} cardinal')
        while True:
            copy = img.copy()
            cv2.rectangle(copy,(x1,y1),(x2,y2),(255,0,255),2)
            text1 = 'Adjust ' + ['top','bottom','left','right'][cardinal]
            text2 = f'< a{" "*20} d >' if cardinal > 1 else f'^ w{" "*14} s v'
            choice = display_preview('confirmBounds',copy,text1, text2, copy=False)
            #print('\t\tSelected ord={}, chr={}.'.format(ord(choice),choice))
            if choice == chr(13): break
            elif choice == chr(27): exit()
            else:
                y1,y2,x1,x2 = adjustBounds(cardinal,choice,y1,y2,x1,x2)

    cv2.destroyAllWindows()

    return [int(i) for i in [y1,y2,x1,x2]]


def adjustBounds(cardinal,c,y1,y2,x1,x2):
    #adjust the bounds
    points = [y1,y2,x1,x2]
    choices = {
        'w':-1,
        'a':-1,
        's':1,
        'd':1
    }

    if c not in choices:
        return points

    points[cardinal] = points[cardinal] + (choices[c]*10)
    print('\t\tNew bounds:{}'.format(points))
    return points


def crop(img):
    #crop the image to the bounds
    y1,y2,x1,x2 = autoBounds(img)
    y1,y2,x1,x2 = confirmBounds(img,y1,y2,x1,x2)
    img = img[y1:y2, x1:x2]
    return img


def saveImg(img,tempfolder,path):
    #saves the img to file
    savepath = str(tempfolder.joinpath(path.name).resolve())
    print(f'\tProcessed {savepath}')
    cv2.imwrite(savepath,img)
    return savepath


def convert_pdf(args,imgs):
    #creates PDF from images
    pw,ph = (595.28, 841.89) #from A4 documentation
    margin = 3.0

    pdf = FPDF('P','mm',(pw,ph))
    pdf.set_margins(margin,margin)
    
    for path,img in imgs.items():
        pdf.add_page()
        pdf.image(path,0,0,*scale_to_page(img,pw,ph,margin))
    
    output_path = os.path.join(args.directory,f'{args.filename}.pdf')
    pdf.output(output_path, "F")
    print(f'\tCreated PDF {output_path}')
    return output_path


def scale_to_page(img, pw, ph, margin):
    #scales the image to the page

    iw = int(img.shape[1])
    ih = int(img.shape[0])

    iar = iw/ih
    par = pw/ph

    if iar > par:
        #w is dominant
        scale = pw/iw
        w = pw
        h = ih * scale
    else:
        #h is dominant
        scale = ph/ih
        h = ph
        w = iw * scale

    return w-(2*margin),h-(2*margin)
if  __name__ =='__main__':start()