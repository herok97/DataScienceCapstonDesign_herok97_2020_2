from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf

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

model = load_model('Classification_Model/save_model/Resnet101_150_by_functional.h5')

def classify(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=150 / img.shape[1], fy=150 / img.shape[0])
    result = model.predict([[img/256]])
    if result < 0:
        return False
    elif result >= 0:
        return True
    else:
        print("예측 결과가 [[1]] 또는 [[0]]이어야 합니다.")