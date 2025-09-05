import keras
import cv2
import argparse
import numpy as np
#from keras.applications.mobilenet_v2 import MobileNetV2 #size=(244, 244)
#from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
#from keras.applications.xception import Xception #size=(299, 299)
#from keras.applications.xception import preprocess_input, decode_predictions
from keras.applications.convnext import ConvNeXtLarge #size=(224, 224)
from keras.applications.convnext import preprocess_input, decode_predictions
from time import sleep
from pygame import mixer

# Cat labels indexed by imagenet synset, with the leading 'n0' removed for integer comparison
cat_labels = {
    2123045, #tabby
    2123159, #tiger
    2123394, #persian
    2123597, #siamese
    2124075, #egyptian
    2125311, #cougar
    2127052, #lynx
}

# Parse arguments
parser = argparse.ArgumentParser(description='Run cat detection software.')
parser.add_argument('--sound', type=str, default="TrainHorn.mp3", help='Path to alarm sound file (default: TrainHorn.mp3)')
args = parser.parse_args()

# Initialize video and audio
video_stream = cv2.VideoCapture(0)
mixer.init()
alarm = mixer.Sound(args.sound)

# Live view window, uncomment 'cv2.imshow()' and 'cv2.destroyWindow()' below to enable
#cv2.namedWindow("LiveView")

if video_stream.isOpened():
    rval, frame = video_stream.read()
else:
    rval = False

processed_input = preprocess_input(keras.utils.img_to_array(frame))
imnet_size = (224, 224)
processed_input = keras.preprocessing.image.smart_resize(processed_input, imnet_size)

# TODO: set include_top=False and retrain the top layers on a smaller number of categories
#       if a retrained model is used, it probably does not need to be as large as ConvNeXtLarge
base_model = ConvNeXtLarge(weights='imagenet')

if rval:
    while True:
        rval, frame = video_stream.read()

        if frame is None:
            break

        #cv2.imshow("LiveView", frame)

        # Preprocessing
        x = keras.preprocessing.image.smart_resize(frame, imnet_size)
        x = preprocess_input(keras.utils.img_to_array(x))
        x = x[np.newaxis, 0:]
        
        predictions = base_model.predict(x)
        predictions = decode_predictions(predictions, top=5)[0]

        #print("predictions: ")
        for pred in predictions:
            #print(pred[1], "w conf: ", pred[2], " id: ", pred[0])

            imnet_id = int(pred[0][2:])
            if imnet_id in cat_labels:
                print("Cat detected!")
                alarm.play()#maxtime=1850)
                break

        # TODO: this does not work unless the live window is open
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            break
        sleep(10)
        

video_stream.release()
#cv2.destroyWindow("LiveView")