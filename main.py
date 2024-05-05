import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
try:
    loaded_model = tf.keras.models.load_model('my_model.keras')
    print('load mo hinh thanh cong')
except (OSError, IOError) as e:
    print('Không thể tải mô hình:', str(e))
    print('Đang huấn luyện một mô hình mới...')
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())

    # MLP with 3 hidden layers
    model.add(tf.keras.layers.Dense(375, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(225, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(135, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit(x_train[..., np.newaxis], tf.keras.utils.to_categorical(y_train), validation_data=(x_test[..., np.newaxis], tf.keras.utils.to_categorical(y_test)), epochs=5, batch_size=256, verbose=2)

    model.save('my_model.keras')
    loaded_model=model
# scores = model.evaluate(x_test, y_test, verbose=0)

# print("DCNN1 Error: %.2f%%" % (100-scores[1]*100))

# loss , accuracy  =model.evaluate(x_test,y_test)
# print(accuracy)
# print(loss)

# for x in range(1, 5):
    # now we are going to read images with OpenCV
while True:
    a = input("Nhập tên ảnh: ")
    img = cv.imread(f'{a}.png')[:, :, 0]  # all of it and 1st and last one
    img = np.invert(np.array([img]))  # invert black to white in images so that model won't get confused
    prediction = loaded_model.predict(img)
    print("----------------")
    print("Giá trị dự đoán là:", np.argmax(prediction))
    print("----------------")
    plt.imshow(img[0], cmap=plt.cm.binary)  # change the color to black and white
    plt.show()