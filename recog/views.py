from django.contrib.staticfiles.storage import staticfiles_storage
from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse
import cv2
path = staticfiles_storage.path('haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier(path)
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(-1)
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()
    def get_frame(self):
        ret,frame= self.video.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        scale_percent = 60 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        ret,jpeg = cv2.imencode('.jpg',frame)
        return jpeg.tobytes()
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request): 
    try:
        return StreamingHttpResponse(gen(VideoCamera()),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted")
        
def home(request):
    return render(request,'recog/home.html')

def index(request):
    return render(request,'recog/index.html')




