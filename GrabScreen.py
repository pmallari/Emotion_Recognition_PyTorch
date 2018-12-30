import numpy as np
import os
from PIL import ImageGrab
import cv2
import time
import torch
from cnn_model import Net
import random
random.seed(42)

# Define Emotion Dictionary
Emotion = {0:'Anger',
           1:'Disgust',
           2:'Fear',
           3:'Happy',
           4:'Sad',
           5:'Surprise',
           6:'Neutral'}

# Process image from BGR to Gray
def process_img(original_image):
	processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	#processed_img = cv2.Canny(processed_img, threshold1 = 100, threshold2 = 200)
	return processed_img

# Convert image to tensors for PyTorch prediction
def image_to_tensor(img):
	img = cv2.resize(img, dsize = (48,48), interpolation = cv2.INTER_CUBIC)
	return torch.Tensor(img.reshape(1,1,48,48))

# Import model architecture and model states
model = Net()
dir_path = os.path.dirname(os.path.realpath(__file__))
model.load_state_dict(torch.load(dir_path + '/models/current_model.pth'))

# Define a starting emotion
curr_emotion = "Neutral"
# Gather emotions and get mode to avoid randomness
emotion_ave = []
# Get current time
last_time = time.time()

while(True):
	#Grabs image from screen
	screen = np.array(ImageGrab.grab(bbox = (300, 300, 500, 500)))
	processed_img = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
	new_screen = process_img(screen)

	# Convert iamge to tensor and predict the emotion
	x = new_screen
	x = image_to_tensor(x)
	output= model(x)
	_, pred = torch.max(output, 1)

	# Gets the mode of the emotion taken within one second
	# This reduces the randomness of the output
	if (time.time() - last_time) > 1:
		curr_emotion = Emotion[max(set(emotion_ave), key = emotion_ave.count)]
		print(curr_emotion)
		emotion_ave = []
		last_time = time.time()
	else:
		emotion_ave.append(int(pred))

	# Show image and prediction
	cv2.putText(processed_img, curr_emotion, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
	cv2.imshow('window2', processed_img)

	#print('Frame rate of {:.3f}s.'.format(1/(time.time() - last_time)))
	#cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)) #Original Screen
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
