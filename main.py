import imutils
from imutils.video import FPS
import numpy as np
import cv2

#image_path = 'Living-Room-Shot.jpg'
prototxt_path = 'models/MobileNetSSD_deploy.prototxt.txt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cap = cv2.VideoCapture(0)
fps = FPS().start()
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=400)

    #image = cv2.imread(image_path)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843,
                                 (300, 300), 127.5)
    net.setInput(blob)
    detected_objects = net.forward()

    for i in np.arange(0, detected_objects.shape[2]):
        confidence = detected_objects[0, 0, i, 2]
        if confidence > min_confidence:
            index = int(detected_objects[0, 0, i, 1])
            box = detected_objects[0, 0, i, 3:7] * np.array([w, h, w, h])
            (upperLeftX, upperLeftY, bottomRightX, bottomRightY) = box.astype("int")

            label = "{}: {:.2f}%".format(classes[index], confidence * 100)
            cv2.rectangle(frame, (upperLeftX, upperLeftY), (bottomRightX, bottomRightY),
                          colors[index], 2)

            cv2.putText(frame, label, (upperLeftX,
                                       upperLeftY - 15 if upperLeftY - 15 > 15 else upperLeftY + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, colors[index], 2)

    cv2.imshow("Object Detection using MobileNetSSD", frame)
    endSession = cv2.waitKey(1) & 0xFF

    if endSession == ord("q"):
        break

    fps.update()
    fps.stop()

cv2.destroyAllWindows()
cap.release()
