#!/usr/bin/python3
!pip install opencv-python
import os
import cv2 as cv
model = tf.keras.models.load_model('tp.model.keras')
image_nb = 1

while os.path.isfile(f"image{image_nb}.png"):
    img_path = f"image{image_nb}.png"
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    resized_img = cv.resize(img, (28 , 28))

    resized_img = np.invert(np.array([resized_img]))
    prediction = model.predict(resized_img)
    print(f"The predicted number is: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    image_nb += 1

