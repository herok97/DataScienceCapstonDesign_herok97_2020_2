import sys
from mtcnn import MTCNN
import tensorflow.compat.v1 as tf
from MTCNN.find_face import find_face
import cv2
tf.logging.set_verbosity(tf.logging.ERROR)
# gpu 오류 해결
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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

    return boxes

def IoU(box1, box2):
    inter = max( min(int(box1[2]), int(box2[2])) - max(int(box1[0]), int(box2[0])), 0) * max( min(int(box1[3]), int(box2[3])) - max(int(box1[1]), int(box2[1])), 0)
    box1_ = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
    box2_ = (int(box2[2]) - int(box2[0])) * (int(box2[3]) - int(box2[1]))
    return inter / (box1_ + box2_ - inter)

def count_TP_FP(predict_boxes, real_boxes):
    TP = 0
    FP = 0

    # 모든 예측한 박스에 대해서
    for predict_box in predict_boxes:
        find = False

        # 모든 실제 박스들과 비교했을 때
        for real_box in real_boxes:

            # IoU가 0.5 이상이면 찾았다고 하고 loop 탈출
            if IoU(predict_box, real_box) > 0.3:
                # print(IoU(predict_box, real_box))
                find = True
                break

        # 예측한 박스와 겹치는 실제 박스를 찾았으면 TP 1 증가, 못찾았으면 엉뚱한 예측을 했으므로 FP 1 증가
        if find:
            TP += 1
        else:
            FP += 1

    return TP, FP


detector = MTCNN()


file = open("./Data/annotation.txt")

# TP_MTCNN = 0
# FP_MTCNN = 0
TOTAL = 0
TP_MYMODEL = 0
FP_MYMODEL = 0
for line in file.readlines():

    splited_line = line.split()
    file_name = splited_line[0]
    real_bbxes = []
    for i in range((len(splited_line)-1)//4):
        bbx = []
        for j in range(4):
            bbx.append(int(splited_line[4*i + (j+1)]))
        real_bbxes.append(bbx)
        TOTAL += 1
    # # 박스 예측하기
    # # pred_boxes = find_boxes("./Data/img/" + file_name + ".jpg")
    # img = cv2.cvtColor(cv2.imread("./Data/img/" + file_name + ".jpg"), cv2.COLOR_BGR2RGB)
    # pred_boxes_MTCNN = [[img['box'][0], img['box'][1], img['box'][0] + img['box'][2], img['box'][1] + img['box'][3] ] for img in detector.detect_faces(img)]
    # # print("real box:", real_bbxes)
    # # print("pred box:", pred_boxes_MTCNN)
    #
    # tp, fp = count_TP_FP(pred_boxes_MTCNN, real_bbxes)
    # TP_MTCNN += tp
    # FP_MTCNN += fp
    #
    # print("TP_MTCNN: ", TP_MTCNN, "FP_MTCNN", FP_MTCNN, "Precision: ", round(TP_MTCNN / float(TP_MTCNN + FP_MTCNN), 2))



    # 박스 예측하기
    pred_boxes_MYMODEL = find_boxes("./Data/img/" + file_name + ".jpg")
    # print("real box:", real_bbxes)
    # print("pred box:", pred_boxes_MYMODEL)

    tp, fp = count_TP_FP(pred_boxes_MYMODEL, real_bbxes)
    TP_MYMODEL += tp
    FP_MYMODEL += fp

    print("TP_MYMODEL: ", TP_MYMODEL, "FP_MYMODEL", FP_MYMODEL, "Precision: ", round(TP_MYMODEL / float(TP_MYMODEL + FP_MYMODEL), 2) , "TOTAL", TOTAL)
