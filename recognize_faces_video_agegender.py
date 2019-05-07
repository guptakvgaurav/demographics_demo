
# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from PIL import Image
import math
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-i", "--input", type=str, 
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

current_milli_time = lambda: int(round(time.time() * 1000))

# Load network
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)


# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
video_path=1
if args["input"] is not None:
    video_path=int(args["input"])

print('Reading video from {}'.format(video_path))
vs = cv2.VideoCapture(video_path)
writer = None

#time.sleep(2.0)
#for i in range(0, 106):
    #rc,fr=vs.read()

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	return_value,frame = vs.read()

	if not return_value: break
	

	#image = Image.fromarray(frame)
        # for store we only want to see part of the frame
	#imagex,imagey = image.size
	#image = image.crop((int(imagex*0.7),int(imagey*.1),int(imagex * .9),int(imagey * .6)))
	#frame = np.array(image)
	frame_w = frame.shape[0] 
	frame_h = frame.shape[1] 


	prev_time = time.time()
	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.api.batch_face_locations([rgb])
	boxes = boxes[0]
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	face_time = current_milli_time() - prev_time


	padding=20
	agegender=[]
	for tbbox in boxes:
		bbox=[tbbox[3],tbbox[0],tbbox[1],tbbox[2]]
		face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
		blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

		gender_pred_st_time = current_milli_time()
		genderNet.setInput(blob)
		genderPreds = genderNet.forward()
		gender = genderList[genderPreds[0].argmax()]
		#print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
		gender_pred_time = current_milli_time() - gender_pred_st_time

		age_pred_st_time = current_milli_time()
		ageNet.setInput(blob)
		agePreds = ageNet.forward()
		age = ageList[agePreds[0].argmax()]
		age_pred_time = current_milli_time() - age_pred_st_time


		label = "Age:{}, {} \n Gender: {}, {}".format(age, str(age_pred_time), gender, str(gender_pred_time))
		#print(label)
		agegender.append(label)

        	# loop over the facial embeddings
	index=0

	rek_st_time = current_milli_time()
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding,0.45)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			print(counts)

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

		# update the list of names
		name = name + "\n" + agegender[index]
		names.append(name)
		index=index+1

	rek_time = current_milli_time() - rek_st_time                

	exec_time = time.time() - prev_time
	info = "%s, %.2f ms" %( 'time', 1000*exec_time)

	# for idx, info in enumerate(infos.split('\n')):
	cv2.putText(frame, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4, color=(0, 255, 0), thickness=2)

                	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		# for idx, val in enumerate(name.split('\n')):
		# 	cv2.putText(frame, val, (left, y-15*idx), cv2.FONT_HERSHEY_SIMPLEX,
		# 		0.45, (0, 255, 0), 1)

		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces to disk
	if writer is not None:
		writer.write(frame)

        	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()



