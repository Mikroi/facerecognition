import cv2
import os

face_cascade = cv2.CascadeClassifier("/Users/jimmy2/PycharmProjects/facerecognition/Files-3/haarcascade_frontalface_default.xml")

this_folder = os.path.dirname(os.path.abspath("/Users/jimmy2/PycharmProjects/facerecognition/Files-3/news.jpg"))
my_file = os.path.join(this_folder, "news.jpg")

img = cv2.imread("/Users/jimmy2/PycharmProjects/facerecognition/Files-3/news.jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img,
scaleFactor = 1.1,
minNeighbors=5)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

print(type(faces))
print(faces)

resized = cv2.resize(img,(int(img.shape[1]/3), int(img.shape[0]/3)))

cv2.imshow("Gray", resized)
cv2.waitKey(0)
cv2.destroyAllWindows('frame')
cv2.waitKey(1)
