import cv2
import sys
import argparse     
import time

from Processer import Processor
"""name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']"""
name_list = ['bicycle', 'motorcycle','childcar','person','doorgap']
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

#from Visualizer import Visualizer
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def cli():
    desc = 'Run TensorRT fall visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-model', help='trt engine file located in ./models', required=False)
    parser.add_argument('-image', help='image file path', required=False)
    args = parser.parse_args()
    model = args.model or 'best.trt'
    img = args.image or 'no.jpg'
    return { 'model': model, 'image': img }

def main():
    # parse arguments
    args = cli()
    
    # setup processor and visualizer
    processor = Processor(model=args['model'])
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test1.mp4")
    count = 0
    while True:
        t_start = time.time()
        ret, img = cap.read()
        #img = img[:, ::-1, :]
        # inference
        if count % 3 ==0:
          output = processor.detect(img) 
        img = cv2.resize(img, (640, 640))
        
        boxes, confs, classes = processor.cls_process(output)
        fps = 1 / (time.time() - t_start)
        #print('fps:', fps)
        #print(boxes, confs, classes)
        if len(boxes) != 0:
            for (f ,conf, cls) in zip(boxes, confs, classes):
                x1 = int(f[0])
                y1 = int(f[1])
                x2 = int(f[2])
                y2 = int(f[3])
                #cls = int(f[-1])
                if int(cls)==1:
                  filename = 'images/'+str(time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))) + '.jpg'
                  cv2.imwrite(filename,img)
                color_lb = compute_color_for_labels(int(cls))
                name = name_list[int(cls)]    
                cv2.rectangle(img, (x1, y1), (x2, y2), color_lb, thickness = 3)
                t_size = cv2.getTextSize(name, 0, fontScale=1, thickness=2)[0]
                cv2.rectangle(img, (x1-1, y1), (x1 + t_size[0] + 34, y1 - t_size[1]), color_lb, -1, cv2.LINE_AA)
                cv2.putText(img, name + str('%.2f'%conf), 
                    (x1, int(y1-2)), 0, 0.8, thickness=2,  
                    color = (255, 255, 255), lineType = cv2.LINE_AA)
                cv2.putText(img, 'FPS: ' + str('%.2f'%fps), (20, 30), 0, 1, thickness = 2, color = (255, 0, 0), lineType = cv2.LINE_AA)
        count +=1
        cv2.imshow("da", img)
        cv2.waitKey(1)
    
    """t_start = time.time()
    img = cv2.imread("535.jpg")
    h,w = img.shape[0],img.shape[1]
    #img = img[:, ::-1, :]
    # inference
    output = processor.detect(img)
    img = cv2.resize(img,(640,640))
    boxes, confs, classes = processor.cls_process(output)
    fps = 1 / (time.time() - t_start)
    #print('fps:', fps)
    #print(boxes, confs, classes)
    if len(boxes) != 0:
      for (f ,conf, cls) in zip(boxes, confs, classes):
        x1 = int(f[0])
        y1 = int(f[1])
        x2 = int(f[2])
        y2 = int(f[3])
        #cls = int(f[-1])
        color_lb = compute_color_for_labels(int(cls))
        name = name_list[int(cls)]    
        cv2.rectangle(img, (x1, y1), (x2, y2), color_lb, thickness = 3)
        t_size = cv2.getTextSize(name, 0, fontScale=1, thickness=2)[0]
        cv2.rectangle(img, (x1-1, y1), (x1 + t_size[0] + 34, y1 - t_size[1]), color_lb, -1, cv2.LINE_AA)
        cv2.putText(img, name + str('%.2f'%conf), (x1, int(y1-2)), 0, 0.8, thickness=2,  
        color = (255, 255, 255), lineType = cv2.LINE_AA)
        cv2.putText(img, 'FPS: ' + str('%.2f'%fps), (20, 30), 0, 1, thickness = 2, color = (255, 0, 0), lineType = cv2.LINE_AA)
    cv2.imshow("da", img)
    cv2.imwrite("test.jpg",img)"""
    # visualizer.draw_results(img, boxes, confs, classes)

if __name__ == '__main__':
    main()   
