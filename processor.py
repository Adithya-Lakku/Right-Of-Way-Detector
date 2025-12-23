import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors,Annotator
from ultralytics.data.augment import LetterBox
import numpy as np


modelDet = YOLO(r"E:\Study\VS_CODE_STUFF\PROJECTS\Right-Of-Way Detector\Veh_pedv8n.pt")  #detection model
modelSeg = YOLO(r"E:\Study\VS_CODE_STUFF\PROJECTS\Right-Of-Way Detector\crosswalkseg_v11n.pt") #segmentation model

testImage = r"E:\Study\VS_CODE_STUFF\PROJECTS\Right-Of-Way Detector\crosswalktestimg.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def getColours(cls_num):
#     base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#     color_index = cls_num % len(base_colors)
#     increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
#     color = [base_colors[color_index][i] + increments[color_index][i] * 
#     (cls_num // len(base_colors)) % 256 for i in range(3)]
#     return tuple(color)



# yolo segmentation

def yolo_seg(img):
    results = modelSeg.predict(source = img,device = device,imgsz = 416,show_boxes = False)

    if results:
        result = results[0]

        if result.masks is not None:
            # masks = result.masks.data.shape[0]
            # n = masks.shape[0]   #yolo gives a tensor as result for masks not np array

            mask = result.masks.data[0].cpu().numpy()  #add all available segments to tensor
            for i in range(1,result.masks.data.shape[0]):
                mask+=result.masks.data[i].cpu().numpy()

            binary_mask = (mask >0.5).astype(np.uint8)  # 1 = segment , 0 = bg
            mask_resized = cv2.resize(binary_mask,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)


            img = result.plot(boxes=False)
            
            extra_data = {"img":img,"framename":"Segmentation","mask":mask_resized}
            
            
            
            return extra_data
        
        else:
            #no masks
            extra_data = {"img":img,"framename":"Segmentation"}
            return extra_data

def yolo_det(img):

    results = modelDet.predict(img,stream=True,device = device,imgsz=416,verbose=True)

    for result in results:
        classes_names = result.names

        for box in results.boxes:
            if box.conf[0] >0.1:

                [x1,y1,x2,y2] = box.xyxy[0]

                x1,y1,x2,y1 = int(x1), int(y1), int(x2), int(y2)

                cls = int(box.cls[0])
                colour = getColours(x1,y1,x2,y2,cls)

                cv2.rectangle(img,(x1,y1),(x2,y2),colour,2)

                cv2.putText(img, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)




def getColours(x1,y1,x2,y2,cls):



class Processor:
    def __init__(self):
        






