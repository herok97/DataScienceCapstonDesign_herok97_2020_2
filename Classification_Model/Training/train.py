from .get_data import get_data
import tensorflow as tf
import matplotlib.pyplot as plt



PATH = '../Data/faces/'
CATEGORIES = ["masked", "no_masked"]
SIZE = 150
IMG_SHAPE = (SIZE, SIZE, 3)

# Fix random seed
tf.random.set_seed(100)

# 데이터 불러오기
X_train, X_test, Y_train, Y_test = get_data(SIZE)

# ResNet101V2 모델 임포트
base_model = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet', input_shape=IMG_SHAPE)

# 베이스 모델 레이어 동결
base_model.trainable = False

# 새 분류 계층 넣기
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

# 콜백 함수 정의
callback_list = [
    tf.keras.callbacks.ModelCheckpoint(  # 에포크마다 현재 가중치를 저장
        filepath="save_file/model_weight.h5",  # 모델 파일 경로
        monitor='accuracy',  # val_loss 가 좋아지지 않으면 모델 파일을 덮어쓰지 않음
        save_best_only=True
    )
]

# 학습 설정 (compile)
base_learing_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learing_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 학습
epochs = 50
batch_size = 128
history = model.fit(X_train, Y_train, epochs=epochs, callbacks=callback_list,
                    batch_size=batch_size, validation_split=0.33, shuffle=True)

#
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

# 학습결과 시각화
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy' + str(SIZE))

plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()