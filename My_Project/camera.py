import cv2
import pandas as pd
class VideoCamera(object):
    stampNumber = 0
    name = ""
    eno = 0

    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        dataset = pd.read_csv('studentTempData.csv')
        values = dataset.iloc[:, :].values
        self.name = values[0][0]
        self.eno = values[0][1]
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):

        success, image = self.video.read()
        face_cascade = cv2.CascadeClassifier('My_Project/cascades/data/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3)
        for (x, y, h, w) in faces:
            self.stampNumber += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 225, 0), 2)
            cv2.imwrite("dataset/ " + self.name + "." + str(self.eno) + '.' + str(self.stampNumber) + ".jpg",
                        gray[y:y + h, x:x + w])
        if cv2.waitKey(20) & 0xFF == ord('q'):
            self.video.release()
            return
        if self.stampNumber > 120:
            self.video.release()
            return

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def recognize(self, recognizer, faceCascade, font, df, col):
        name=""
        Id=-1
        success, image = self.video.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf <= 100.00 and Id > 0:
                name = df.loc[df["eno"] == int(Id)]
                name = name.iloc[0]['name']
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 260, 0), 7)
                cv2.putText(image, str(conf), (x, y - 40), font, 2, (255, 255, 255), 3)
            elif conf > 100.00:
                name = "unknown"
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 260, 0), 7)
                cv2.putText(image, str("Unknown"), (x, y - 40), font, 2, (255, 255, 255), 3)
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes(), not name in col, name , Id