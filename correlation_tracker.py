import ast
import cv2
import dlib
import json
import os
import sys
import shutil
import zipfile


current_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

zipfile_name = sys.argv[1]
output_filename = 'correlation_tracking_output'
archive = zipfile.ZipFile(zipfile_name, 'r')
archive.extractall(output_filename)
archive.close()

zip_output = zipfile.ZipFile(output_filename + '.zip', 'w')

positions = sys.argv[2]
positions = list(ast.literal_eval(positions))
num_faces = len(positions)

started_tracking = False

trackers = []
for i in range(num_faces):
	trackers.append(dlib.correlation_tracker())

for f in sorted(os.listdir(output_filename)):
	filename = output_filename + '/' + f
	frame = cv2.imread(filename)
	height, width, channels = frame.shape

   	if not started_tracking:
		for i in range(num_faces):
			position = positions[i]
			tracker = trackers[i]
			detection = dlib.drectangle(position[0], position[1], position[2], position[3])
			tracker.start_track(frame, detection)
		started_tracking = True
   	else:
		for i in range(num_faces):
			trackers[i].update(frame)

	for i in range(num_faces):
		rect = trackers[i].get_position()
		try:
			cv2.rectangle(frame, (int(rect.left()), int(rect.top())), (int(rect.right()), int(rect.bottom())), (255, 0, 0), 2)
		except OverflowError:
			continue

	# do something with the aligned faces here
	cv2.imwrite(filename, frame)
	zip_output.write(filename, f)

zip_output.close()
shutil.rmtree(output_filename)

print current_directory + output_filename + '.zip'
