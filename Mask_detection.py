import cv2
import numpy as np
import keras

cam = cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")

model=keras.models.load_model("Model/mask_detection.h5")
while True:
    X = []

    check, frame = cam.read()
    frame = cv2.flip(frame, 2, 2)
    frame = cv2.resize(frame, (500, 500))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # (thresh, black_and_white) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = cv2.resize(frame[y:y + h, x:x + w], (160, 160))
        X.append(roi_color)
        X_np=np.array(X)
        X_scale=X_np/255
        re = np.argmax(model.predict(X_scale)[0])
        if re==0:
            cv2.putText(frame,"NO MASK",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        else:
            cv2.putText(frame, "MASK", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('video', frame)


    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()


# X=[]
# img=cv2.imread("Images/people.jpeg")
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# face_cascade=cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")
# faces=face_cascade.detectMultiScale(gray,1.3,5)
# print(len(faces))
# if len(faces)>0:
#     (x,y,w,h)=faces[0]
#     roi_color = cv2.resize(img[y:y+h, x:x+w],(160,160))
#     X.append(roi_color)
# X=np.array(X)
# X_scale=X/255
#
# model=keras.models.load_model("Model/mask_detection.h5")
# re=np.argmax(model.predict(X_scale)[0])




