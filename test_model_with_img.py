import sys
sys.path.append('../')
from MTCNN.find_face import find_face
from Classification_Model.classify_face import classify
import cv2

def find_boxes(img_path):

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    faces = find_face(img)
    boxes = []
    for bbox in faces:
        bbox[0] = max(1, bbox[0])
        bbox[1] = max(1, bbox[1])
        bbox[2] = min(img.shape[1]-1, bbox[2])
        bbox[3] = min(img.shape[0]-1, bbox[3])
        boxes.append(bbox)
    return img, boxes

def show_boxes(img_path):
    img, boxes = find_boxes(img_path)

    for bbox in boxes:
        face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if classify(face):
            print("마스크를 착용하지 않은 사람이 있습니다.")
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (255, 0, 0), 2)
        elif not classify(face):
            print("마스크를 착용한 사람이 있습니다.")
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('detected_' + img_path, img)


# print(find_boxes('test_img.jpg'))
show_boxes('1231.jpg')