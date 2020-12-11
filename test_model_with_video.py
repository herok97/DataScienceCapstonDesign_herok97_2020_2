import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import tensorflow as tf
from MTCNN.find_face import find_face

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

model = load_model('Classification_Model/save_file/Resnet101_150_by_functional.h5')


def classify(img):
    img = cv2.resize(img, None, fx=150 / img.shape[1], fy=150 / img.shape[0])
    result = model.predict([[img / 256]])
    if result < 0:
        return False
    elif result >= 0:
        return True
    else:
        print("예측 결과가 [[1]] 또는 [[0]]이어야 합니다.")


detector = MTCNN()


def find_masked_vs_no_masked(img):
    # 내가 만든 MTCNN 모듈 이용
    faces = find_face(img, threshold=[0.9, 0.9, 0.9], factor=0.7, minsize=50)

    # 기존 MTCNN 모듈 이용
    # faces = [(face['box'], face['confidence']) for face in detector.detect_faces(img)]

    for i in range(len(faces)):

        # 기존 MTCNN 모듈 이용
        bounding_box = faces[i][0]
        bounding_box[0] = max(1, bounding_box[0])
        bounding_box[1] = max(1, bounding_box[1])
        bounding_box[2] = max(1, bounding_box[2])
        bounding_box[3] = max(1, bounding_box[3])

        face = img[bounding_box[1]:bounding_box[1] + bounding_box[3],
               bounding_box[0]:bounding_box[0] + bounding_box[2]]

        # 내가 만든 MTCNN 모듈 이용
        # bounding_box = faces[i]
        # bounding_box[0] = max(1, bounding_box[0])
        # bounding_box[1] = max(1, bounding_box[1])
        # bounding_box[2] = max(1, bounding_box[2])
        # bounding_box[3] = max(1, bounding_box[3])
        #
        # face = img[bounding_box[1]:bounding_box[3],
        #        bounding_box[0]:bounding_box[2]]

        if classify(face):
            print("마스크를 착용하지 않은 사람이 있습니다.")
            cv2.rectangle(img, (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (255, 0, 0), 1)
        elif not classify(face):
            print("마스크를 착용한 사람이 있습니다.")
            cv2.rectangle(img, (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 255, 0), 1)

    return img


def make_test_video(filename):
    # 재생할 파일
    VIDEO_FILE_PATH = filename

    # 동영상 파일 열기
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)

    # 잘 열렸는지 확인
    if cap.isOpened() == False:
        print('Can\'t open the video (%d)' % (VIDEO_FILE_PATH))
        exit()

    # 재생할 파일의 넓이 얻기
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 재생할 파일의 높이 얻기
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # 재생할 파일의 프레임 레이트 얻기
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('width {0}, height {1}, fps {2}'.format(width, height, fps))

    # XVID가 제일 낫다고 함.
    # linux 계열 DIVX, XVID, MJPG, X264, WMV1, WMV2.
    # windows 계열 DIVX
    # 저장할 비디오 코덱
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # 저장할 파일 이름
    save_filename = "detected" + filename

    # 파일 stream 생성
    out = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
    # filename : 파일 이름
    # fourcc : 코덱
    # fps : 초당 프레임 수
    # width : 넓이
    # height : 높이

    # 얼굴 인식용
    frames = 1
    import numpy as np
    while (True):
        # 파일로 부터 이미지 얻기
        ret, frame = cap.read()
        # 더 이상 이미지가 없으면 종료
        # 재생 다 됨
        if frame is None:
            break;

        # 얼굴인식 영상 처리
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = find_masked_vs_no_masked(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 인식된 이미지 파일로 저장
        out.write(img)
        print("진행률:", str(round(float(frames) / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100, 2)) + '%')
        frames += 1

    # 재생 파일 종료
    cap.release()
    # 저장 파일 종료
    out.release()

