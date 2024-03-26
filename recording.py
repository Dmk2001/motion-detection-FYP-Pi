import os
import queue
import shutil
import threading
import time
from uuid import uuid4
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import datetime
import numpy as np
from picamera2 import Picamera2
from picamera2.previews.qt import QGlPicamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
from gpiozero import CPUTemperature
from firebase_admin import credentials, firestore
import firebase_admin
from firebase_admin import credentials, storage, firestore
import cv2
from skimage.metrics import mean_squared_error as ssim


cred = credentials.Certificate("key.json")
firebase_app = firebase_admin.initialize_app(
    cred, {"storageBucket": "motion-detection-fyp.appspot.com"}
)
db = firestore.client()
recording = False
motion = False
loop = True


q = queue.Queue()


class PiSettings:
    def __init__(self):
        self.video_resolution = []
        self.recording_length = 0
        self.movementThreshold = 0
piSettings = PiSettings()

def on_snapshot(doc_snapshot, changes, real_time):
    global old_resolution
    print(doc_snapshot)

    resolution = doc_snapshot[3].to_dict().get("resolution")

    print(resolution)

    if resolution == 480:
        piSettings.video_resolution = [640, 480]
    elif resolution == 720:
        piSettings.video_resolution = [1280, 720]
    piSettings.recording_length = doc_snapshot[0].to_dict().get("recording_length")
    piSettings.movementThreshold = doc_snapshot[0].to_dict().get("movement_threshold")

    for change in changes:
        
        if (
            change.type.name == "MODIFIED"
            and "capture_photo" in change.document.to_dict()
        ):
            if change.document.get("capture_photo") == True:
                print("Taking Photo")
                on_photo_button_clicked()
                time.sleep(1)
                db.collection("settings").document("record").set(
                    {"capture_photo": False}
                )

        if (
            change.type.name == "MODIFIED"
            and "capture_video" in change.document.to_dict()
        ):
            if change.document.get("capture_video") == True:
                print("Capture Video. Length = " + str(piSettings.recording_length))
                record_video()
                time.sleep(1)
                db.collection("settings").document("record_video").set(
                    {"capture_video": False}
                )
        if (
            change.type.name == "MODIFIED"
            and "resolution" in change.document.to_dict()
        ):
            
            print("Rebooting")
            os.system("sudo reboot")
            

    callback_done.set()



callback_done = threading.Event()
doc_ref = db.collection("settings")


doc_watch = doc_ref.on_snapshot(on_snapshot)


def update_pi_status():
    cpu = CPUTemperature() 
    while True:

        db.collection("status").document("pi_status").set(
            {
                "cpu_temp": str(cpu.temperature) + "C",
                "recording": recording,
                "last_updated": str(datetime.datetime.now()),
            }
        )
        time.sleep(5)


status_thread = threading.Thread(target=update_pi_status)
status_thread.start()


picam2 = Picamera2()
time.sleep(5)

picam2.configure(
    picam2.create_video_configuration(
        main={"size": (piSettings.video_resolution[0], piSettings.video_resolution[1])},
        controls={"FrameRate": 30},
    )
)
app = QApplication([])


def receive_frames():
    while loop:
        if motion == True:
            frame = picam2.capture_array()
            q.put(frame)


receive_thread = threading.Thread(target=receive_frames)
receive_thread.start()


def motion_detection():
    while loop:
        global old_frame
        global motion
        global recording
        global activity_count
        if motion == True:

            if q.empty() != True:
                frame = q.get()
                resized_frame = cv2.resize(frame, resized_video)
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                final_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

                diff = cv2.absdiff(final_frame, old_frame)
                result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
                ssim_val = int(ssim(result, blank))
                old_frame = final_frame
                print("Motion: " + str(ssim_val))

                # count the number of frames where the ssim value exceeds the threshold value, and begin
                # recording if the number of frames exceeds start_frames value
                if not recording:
                    if ssim_val > piSettings.movementThreshold:
                        activity_count += 1
                        if activity_count >= 10:
                            if not testing:
                                folderdate = datetime.datetime.now().strftime(
                                    "%Y-%m-%d"
                                )
                                picam2.stop()
                                picam2.start()
                                video_button.setEnabled(False)
                                encoder = H264Encoder(bitrate=10000000)
                                input = time.strftime("%Y-%m-%d-%H%M%S")
                                input_file = FileOutput(input + ".h264")
                                picam2.start_encoder(encoder, input_file)
                                print("Started Recording")

                                start_time = time.time()
                            else:
                                print(0)
                                # print(filedate + " recording started - Testing mode")
                            recording = True
                            activity_count = 0
                    else:
                        activity_count = 0

                # if already recording, count the number of frames where there's no motion activity and stop
                # recording if it exceeds the tail_length value
                else:
                    if (time.time() - start_time) > piSettings.recording_length:
                        picam2.stop_encoder()
                        print("Recording Stopped")
                        shutil.move(
                            input + ".h264",
                            os.getcwd() + "/ConversionQueue/",
                            copy_function=shutil.copy2,
                        )
                        picam2.stop()
                        picam2.start()
                        recording = False
                    if (time.time() - start_time) < piSettings.recording_length:
                        if ssim_val < piSettings.movementThreshold:
                            activity_count += 1
                            if activity_count >= 10:

                                if not testing:
                                    picam2.stop_encoder()
                                    print("Recording Stopped")
                                    shutil.move(
                                        input + ".h264",
                                        os.getcwd() + "/ConversionQueue/",
                                        copy_function=shutil.copy2,
                                    )
                                    # delete recording if total length is equal to the tail_length value,
                                    # indicating a false positive
                                    if auto_delete:
                                        recorded_file = cv2.VideoCapture()
                                        recorded_frames = recorded_file.get(
                                            cv2.CAP_PROP_FRAME_COUNT
                                        )
                                        if recorded_frames < 5 + (
                                            30 / 2
                                        ) and os.path.isfile(input_file):
                                            os.remove(input_file)
                                            print("auto-deleted")
                                else:
                                    print(0)
                                    # print(" recording stopped - Testing mode")
                                picam2.stop()
                                picam2.start()
                                recording = False

                        else:
                            activity_count = 0

            else:
                time.sleep(1)


motion_thread = threading.Thread(target=motion_detection)
motion_thread.start()


def record_video():
    global recording
    global file_output
    global input
    recordinglength = piSettings.recording_length
    if not recording:
        picam2.stop()
        picam2.start()
        video_button.setEnabled(False)
        encoder = H264Encoder(bitrate=10000000)
        input = time.strftime("%Y-%m-%d-%H%M%S")
        input_file = FileOutput(input + ".h264")
        picam2.start_encoder(encoder, input_file)
        recording = True
        time.sleep(recordinglength)
        picam2.stop_encoder()
        video_button.setText("Start recording")
        recording = False
        time.sleep(2)
        shutil.move(
            input + ".h264",
            os.getcwd() + "/ConversionQueue/",
            copy_function=shutil.copy2,
        )
        picam2.stop()
        video_button.setEnabled(True)
        picam2.start()


def on_motion_button_clicked():
    global motion
    if motion == True:
        motion = False
        print("Motion = " + str(motion))
        motion_button.setText("Enable Motion Detection")
    else:
        motion = True
        print("Motion = " + str(motion))
        motion_button.setText("Disable Motion Detection")


def on_photo_button_clicked():
    global recording
    global taking_photo
    if not recording:
        taking_photo = True
        photo_button.setEnabled(False)
        cfg = picam2.create_still_configuration()
        picam2.switch_mode_and_capture_file(cfg, "test.jpeg")
        bucket = storage.bucket()

        blob = bucket.blob("test.jpeg")

        token = uuid4()
        metadata = {"firebaseStorageDownloadTokens": token}
        blob.metadata = metadata

        blob.upload_from_filename("test.jpeg")
        blob.make_public()
        gcs_storageURL = blob.public_url

        firebase_storageURL = "https://firebasestorage.googleapis.com/v0/b/{}/o/{}?alt=media&token={}".format(
            "motion-detection-fyp.appspot.com", "test.jpeg", token
        )
        db.collection("photo").document("test.jpeg").set(
            {
                "URL": firebase_storageURL,
                "uploadTimestamp": str(datetime.datetime.now()),
            }
        )
        db.collection("settings").document("record").set({"capture_photo": False})
        taking_photo = False
        photo_button.setEnabled(True)


# qpicamera2 = QGlPicamera2(
#     picam2,
#     width=piSettings.video_resolution[0],
#     height=piSettings.video_resolution[1],
#     keep_ar=False,
# )
video_button = QPushButton("Record " + str(piSettings.recording_length) + "s of Video")
motion_button = QPushButton("Enable Motion Detection")
photo_button = QPushButton("Click to capture JPEG")
window = QWidget()


photo_button.clicked.connect(on_photo_button_clicked)
motion_button.clicked.connect(on_motion_button_clicked)
video_button.clicked.connect(record_video)
# qpicamera2.done_signal.connect(capture_done)
picam2.start()

frame = np.zeros((100, 100, 3), dtype=np.uint8)
activity_count = 0
resized_video = (256, 144)
resized_frame = cv2.resize(frame, resized_video)
gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
old_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
blank = np.zeros((resized_video[1], resized_video[0]), np.uint8)
recording = False
taking_photo = False
testing = False
auto_delete = False
layout_h = QHBoxLayout()
layout_v = QVBoxLayout()

layout_v.addWidget(video_button)
layout_v.addWidget(photo_button)
layout_v.addWidget(motion_button)

# layout_h.addWidget(qpicamera2, 80)
layout_h.addLayout(layout_v, 20)
window.setWindowTitle("Qt Picamera2 App")
window.resize(1200, 600)
window.setLayout(layout_h)

window.show()
app.exec()
