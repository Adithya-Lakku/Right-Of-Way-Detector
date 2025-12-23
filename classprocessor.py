import cv2
import torch
import mss
from ultralytics import YOLO
from ultralytics.utils.plotting import colors,Annotator
from ultralytics.data.augment import LetterBox
import numpy as np
import threading
import time

STATE_RED = 1
STATE_GREEN = 0



# def getColours(cls_num):
#     base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#     color_index = cls_num % len(base_colors)
#     increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
#     color = [base_colors[color_index][i] + increments[color_index][i] *
#     (cls_num // len(base_colors)) % 256 for i in range(3)]
#     return tuple(color)

class Processor:

    def __init__(self):
        self.modelDet = YOLO(r"E:\Study\VS_CODE_STUFF\PROJECTS\Right-Of-Way Detector\Veh_pedv8n.pt")  #detection model
        self.modelSeg = YOLO(r"E:\Study\VS_CODE_STUFF\PROJECTS\Right-Of-Way Detector\crosswalkseg_v11n.pt") #segmentation model

        self.testImage = r"E:\Study\VS_CODE_STUFF\PROJECTS\Right-Of-Way Detector\crosswalktestimg.jpg"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state = STATE_RED # 0 is green , 1 is red

        self.mask = None

        #only temporary

        self.mode=0 # 0 - seg, 1- det


    def yolo_seg(self,img):
        results = self.modelSeg.predict(source = img,device = self.device,imgsz = 416,show_boxes = False)

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
                self.mask = mask_resized



                return extra_data

            else:
                #no masks
                extra_data = {"img":img,"framename":"Segmentation"}
                return extra_data


    def yolo_det(self,img):

        results = self.modelDet.predict(img,stream=True,device = self.device,imgsz=416,verbose=True)

        for result in results:
            classes_names = result.names

            for box in result.boxes:
                if box.conf[0] >0.7:

                    [x1,y1,x2,y2] = box.xyxy[0]

                    x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])
                    class_name = classes_names[cls]
                    print(class_name)
                    print("cls is")
                    print(cls)
                    print("\n\n")
                    print(box)

                    
                    colour = self.getColours(x1,y1,x2,y2,cls)

                    cv2.rectangle(img,(x1,y1),(x2,y2),colour,2)

                    cv2.putText(img, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

        return img


    def getColours(self, x1, y1, x2, y2, cls): #functuon to get colors
        red = (0, 0, 255)
        green = (0, 255, 0)
        white = (255, 255, 255)

        car = 1 # for time being consider !ped = vehicle
        ped = 2



        if(cls ==ped):
            if(self.onSeg(x1,x2,y2)):
                if(self.state == STATE_RED):
                    return red
                else:
                    return green
            else:
                return green
       
        if(cls !=ped):
            if(self.onSeg(x1,x2,y2)):
                if(self.state == STATE_RED):
                    return green
                else:
                    return red
            else:
                return green



        
        # if cls == ped and self.onSeg(x1, x2, y2):
        #     return red if self.state == STATE_RED else green

        # if cls == car and self.onSeg(x1, x2, y2):
        #     return red if self.state == STATE_GREEN else green

        return white #default color


    def onSeg(self, x1, x2, y2): #funtion to check if on mask
        if self.mask is None:
            return False

        cx = (x1 + x2) // 2
        h, w = self.mask.shape

        if not (0 <= y2 < h and 0 <= cx < w):  #check used to make sure mask doesnt go out of picture
            return False

        return self.mask[y2, cx] == 1
    
    def start_segmentation(self):
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 400, "height": 400}

            while True:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
               
                extra_data = self.yolo_seg(img)

                
                        
                
                cv2.imshow(extra_data["framename"], extra_data["img"])
                    
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

    def detectfunction(self):
        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 400, "height": 400}

            while True:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
               
               
                
                
                
                img = self.yolo_det(img)

                if(self.state ==1 ):
                    clr = (0,0,255)
                else:
                    clr = (0,255,0)
                        
                cv2.circle(img, center=(20,20), radius=20, color=clr, thickness=-1)
                cv2.imshow("detection", img)
                    
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break



    

#img = cv2.imread(r"E:\Study\VS_CODE_STUFF\PROJECTS\Right-Of-Way Detector\crosswalktestimg.jpg")

def startlive(processor):
       
        

        with mss.mss() as sct:
            monitor = {"top": 0, "left": 0, "width": 400, "height": 400}

            while True:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                processor.yolo_seg(img)
               
                # if(processor.mode == 0):
                #     processor.yolo_seg(img)
                #     processor.mode = 1
                
                img = processor.yolo_det(img)

                if(processor.state ==1 ):
                    clr = (0,0,255)
                else:
                    clr = (0,255,0)
                        
                cv2.circle(img, center=(20,20), radius=20, color=clr, thickness=-1)
                cv2.imshow("detection", img)
                    
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break




def traffic_timer(processor, red_time=5, green_time=5):
    while True:
        processor.state = STATE_RED
        print("STATE: RED")
        time.sleep(red_time)

        processor.state = STATE_GREEN
        print("STATE: GREEN")
        time.sleep(green_time)



 
#creating an object here just for testing


def start_detection(processor):



    t1 = threading.Thread(target=traffic_timer,args=(processor,),daemon=True)
    t2 = threading.Thread(target=processor.detectfunction,daemon=True)

    t1.start()
    t2.start()


    while True:
        time.sleep(1)









#cv2.imshow(extra_data["framename"],extra_data["img"])
#cv2.setMouseCallback(extra_data["framename"],click_event,param = extra_data)

#cv2.destroyAllWindows()

# img = processor.yolo_det(img)



# cv2.imshow("detection",img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

















