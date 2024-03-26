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


class PiSettings:

    def __init__(self):
        self.detections = 0
        self.video_resolution = []
        self.annotation = True
        self.numThreads = 2





def on_snapshot(doc_snapshot, changes, real_time):
    # for change in changes:
    #     if change.type.name == "MODIFIED":
    #         # print(f"Modified setting: {change.document.to_dict().get("resolution")}")
    
    max_results = doc_snapshot[0].to_dict().get(DETECTIONS)
    annotation = doc_snapshot[0].to_dict().get(ANNOTATION)
        
    piSettings.annotation = annotation
    piSettings.detections = max_results
    resolution = doc_snapshot[3].to_dict().get(RESOLUTION)
    print(piSettings.annotation)
    if resolution == 480:
        piSettings.video_resolution = [640, 480]
    elif resolution == 720:
        piSettings.video_resolution = [1280, 720]
    callback_done.set()

callback_done = threading.Event()

piSettings = PiSettings()
doc_ref = db.collection("settings")
doc_watch = doc_ref.on_snapshot(on_snapshot)


def upload(fileToUpload, detections, videoResolution):

    print(fileToUpload)

    try:
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
            firebase_storageURL = "https://firebasestorage.googleapis.com/v0/b/{}/o/{}?alt=media&token={}".format(
                "motion-detection-fyp.appspot.com",
                fileToUpload.split("Output")[1],
                token,
            )
            db.collection("videos").document(fileToUpload.split("Output")[1]).set(
                {
                    "URL": firebase_storageURL,
                    "uploadTimestamp": str(datetime.datetime.now()),
                    "timestamp": fileToUpload.split("Output")[1].split(".mp4")[0],
                    DETECTIONS: detections,
                    RESOLUTION: videoResolution,
                }
            )
        else:
            firebase_storageURL = "https://firebasestorage.googleapis.com/v0/b/{}/o/{}?alt=media&token={}".format(
                "motion-detection-fyp.appspot.com",
                "2" + fileToUpload.split("/2")[1],
                token,
            )
            db.collection("videos").document("2" + fileToUpload.split("/2")[1]).set(
                {
                    "URL": firebase_storageURL,
                    "uploadTimestamp": datetime.datetime.now(),
                    "timestamp": "2" + fileToUpload.split("/2")[1].split(".mp4")[0],
                    DETECTIONS: detections,
                    RESOLUTION: videoResolution,
                }
            )

        os.remove(fileToUpload)
    except:
        print("Failed")


def conversionQueue():
    while True:
        time.sleep(5)
        print("Looking for videos to convert ")
        if os.listdir(CONVERSION_PATH) != []:
            fileToConvert = CONVERSION_PATH + os.listdir(CONVERSION_PATH)[0]
            outputFile = fileToConvert.split(".")[0] + ".mp4"
            if len(outputFile.split("Output")) == 2:

                shutil.move(
                    fileToConvert,
                    os.getcwd() + "/UploadQueue/",
                    copy_function=shutil.copy2,
                )
            else:
                convert_video(fileToConvert, outputFile)
                shutil.move(
                    outputFile,
                    ANNOTATION_PATH,
                    copy_function=shutil.copy2,
                )
        time.sleep(10)


conversion_thread = threading.Thread(target=conversionQueue)
conversion_thread.start()


def convert_video(input, output):
    try:
        print(input)
        print(output)
        subprocess.run(["MP4Box", "-add", input, output], check=True)
        os.remove(input)
    except subprocess.CalledProcessError as e:
        print(e)


while True:
    time.sleep(5)
    # print(os.listdir(os.getcwd() + "/output"))
    if os.listdir(ANNOTATION_PATH) != []:
        input = os.listdir(ANNOTATION_PATH)[0]

        cap = cv2.VideoCapture(ANNOTATION_PATH + input.split(".")[0] + ".mp4")
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if piSettings.annotation == True:

            base_option = core.BaseOptions(
                file_name=model, use_coral=False, num_threads=piSettings.numThreads
            )
            detection_options = processor.DetectionOptions(
                max_results=piSettings.detections, score_threshold=0.3
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
                (piSettings.video_resolution[0], piSettings.video_resolution[1]),
            )
            detector = vision.ObjectDetector.create_from_options(options)
            detection_set = set(())
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
                cv2.imshow("", frame)
                if cv2.waitKey(30) == ord("q"):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            print(outputFile)
            upload(outputFile, detection_set, piSettings.video_resolution)
            os.remove(ANNOTATION_PATH + input)

            time.sleep(5)
        else:

            print("Annotating off")
            upload(ANNOTATION_PATH + input, None, piSettings.video_resolution)

            time.sleep(5)
    else:
        time.sleep(5)
