import datetime
import os
import shutil
import subprocess
import threading
import time
from uuid import uuid4
import cv2
from tflite_support.task import core, processor, vision
import utils

from firebase_admin import credentials, storage, firestore
import firebase_admin
from urllib3.exceptions import ProtocolError
from google.api_core import retry

predicate = retry.if_exception_type(
    ConnectionResetError, ProtocolError
)
reset_retry = retry.Retry(predicate)

ANNOTATION_PATH = os.getcwd() + "/AnnotationQueue/"
CONVERSION_PATH = os.getcwd() + "/ConversionQueue/"
DETECTIONS = "detections"
RESOLUTION = "resolution"
ANNOTATION = "annotation"
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(
    cred, {"storageBucket": "motion-detection-fyp.appspot.com"}
)
db = firestore.client()
model = "efficientdet_lite0.tflite"
converting = False
annotating = False


class PiSettings:

    def __init__(self):
        self.detections = 0
        self.annotation = True
        self.numThreads = 3

    def toString(self):
        string = "Annotating set to {}. Number of Detections set to {}. Number of threads to use: {}".format(
            self.annotation, self.detections, self.numThreads
        )
        return string


def on_snapshot(doc_snapshot, changes, real_time):

    max_results = doc_snapshot[0].to_dict().get(DETECTIONS)
    annotation = doc_snapshot[0].to_dict().get(ANNOTATION)

    piSettings.annotation = annotation
    piSettings.detections = max_results

    print(piSettings.toString())

    callback_done.set()


callback_done = threading.Event()

piSettings = PiSettings()
doc_ref = db.collection("settings")
doc_watch = doc_ref.on_snapshot(on_snapshot)


def upload(fileToUpload, detections, videoResolution):

    print("Uploading File: " + fileToUpload)

    # try:    
    bucket = storage.bucket()
    if "Output" in fileToUpload:
        blob = bucket.blob(fileToUpload.split("Output")[1])
    else:
        blob = bucket.blob("2" + fileToUpload.split("/2")[1])
    token = uuid4()
    metadata = {"firebaseStorageDownloadTokens": token}
    blob.metadata = metadata

    blob.upload_from_filename(fileToUpload)
    blob.make_public()
    gcs_storageURL = blob.public_url
    if ("Output") in fileToUpload:
        timestamp = datetime.datetime.strptime(fileToUpload.split("Output")[1].split(".mp4")[0], "%Y-%m-%d-%H%M%S")
        firebase_storageURL = "https://firebasestorage.googleapis.com/v0/b/{}/o/{}?alt=media&token={}".format(
            "motion-detection-fyp.appspot.com",
            fileToUpload.split("Output")[1],
            token,
        )
        db.collection("videos").document(fileToUpload.split("Output")[1]).set(
            {
                "URL": firebase_storageURL,
                "uploadTimestamp": str(datetime.datetime.now()),
                "timestamp": timestamp,
                DETECTIONS: detections,
                RESOLUTION: videoResolution,
            }
        )
    else:
        timestamp = datetime.datetime.strptime("2" + fileToUpload.split("/2")[1].split(".mp4")[0], "%Y-%m-%d-%H%M%S")
        firebase_storageURL = "https://firebasestorage.googleapis.com/v0/b/{}/o/{}?alt=media&token={}".format(
            "motion-detection-fyp.appspot.com",
            "2" + fileToUpload.split("/2")[1],
            token,
        )
        db.collection("videos").document("2" + fileToUpload.split("/2")[1]).set(
            {
                "URL": firebase_storageURL,
                "uploadTimestamp": str(datetime.datetime.now()),
                "timestamp": timestamp,
                DETECTIONS: detections,
                RESOLUTION: videoResolution,
            }
        )
    print("File uploaded to: " + firebase_storageURL)
    os.remove(fileToUpload)
    # except:
    #     cred = credentials.Certificate("key.json")
        
    #     db = firestore.client()
    #     upload(fileToUpload, detections, videoResolution)

    


def conversionQueue():
    global annotating
    global converting
    time.sleep(5)
    while True:

        if annotating == False:
            if os.listdir(CONVERSION_PATH) != []:
                converting = True
                fileToConvert = CONVERSION_PATH + os.listdir(CONVERSION_PATH)[0]
                outputFile = fileToConvert.split(".")[0] + ".mp4"

                convert_video(fileToConvert, outputFile)
                shutil.move(
                    outputFile,
                    ANNOTATION_PATH,
                    copy_function=shutil.copy2,
                )
                converting = False
            time.sleep(15)

        else:
            print("Will convert after annotation")
            time.sleep(15)


conversion_thread = threading.Thread(target=conversionQueue)
conversion_thread.start()


def convert_video(input, output):
    try:
        print("Converting " + input + " to " + output)
        subprocess.run(["MP4Box", "-add", input, output], check=True)
        os.remove(input)
    except subprocess.CalledProcessError as e:
        print("Failed to Convert")
        print(e)


while True:

    time.sleep(5)
    # print(os.listdir(os.getcwd() + "/output"))
    if converting == False:
        if os.listdir(ANNOTATION_PATH) != []:
            annotating = True
            input = os.listdir(ANNOTATION_PATH)[0]

            cap = cv2.VideoCapture(ANNOTATION_PATH + input.split(".")[0] + ".mp4")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = [width, height]
            if piSettings.annotation == True:

                base_option = core.BaseOptions(
                    file_name=model, use_coral=False, num_threads=piSettings.numThreads
                )
                detection_options = processor.DetectionOptions(
                    max_results=piSettings.detections, score_threshold=0.5
                )
                options = vision.ObjectDetectorOptions(
                    base_options=base_option, detection_options=detection_options
                )
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                outputFile = ANNOTATION_PATH + "Output" + input
                out = cv2.VideoWriter(
                    outputFile,
                    fourcc,
                    fps,
                    (resolution[0], resolution[1]),
                )
                detector = vision.ObjectDetector.create_from_options(options)
                detection_set = set(())

                pos = (20, 60)
                height = 1.5
                weight = 2
                colour = (255, 0, 0)
                fps = 0
                tStart = time.time()
                print("Annotating File: " + input)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting...")
                        break
                    imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imTensor = vision.TensorImage.create_from_array(frame)
                    detections = detector.detect(imTensor)

                    detectionResult = detections.detections

                    for item in detectionResult:

                        category_name = item.categories[0].category_name
                        detection_set.add(category_name)

                    image = utils.visualize(frame, detections)

                    out.write(frame)
                    cv2.putText(
                        frame,
                        str(int(fps)) + " FPS",
                        pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        height,
                        colour,
                        weight,
                    )
                    cv2.imshow(str(resolution), frame)
                    if cv2.waitKey(30) == ord("q"):
                        break
                    tEnd = time.time()
                    loopTime = tEnd - tStart
                    fps = 0.9 * fps + 0.1 * 1 / loopTime
                    print(fps)
                    tStart = time.time()

                cap.release()
                out.release()
                cv2.destroyAllWindows()

                upload(outputFile, detection_set, resolution)
                os.remove(ANNOTATION_PATH + input)
                annotating = False
                time.sleep(5)
            else:

                print("Annotating off")
                upload(ANNOTATION_PATH + input, None, resolution)
                annotating = False
                time.sleep(5)
        else:
            time.sleep(5)
