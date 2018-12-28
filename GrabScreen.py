import numpy as np
from PIL import ImageGrab, ImageDraw, ImageFont
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_model import Net


Emotion = {0:'Anger',
           1:'Disgust',
           2:'Fear',
           3:'Happy',
           4:'Sad',
           5:'Surprise',
           6:'Neutral'}


def process_img(original_image):
	processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	#processed_img = cv2.Canny(processed_img, threshold1 = 100, threshold2 = 200)
	return processed_img

def image_to_tensor(img):
	img = cv2.resize(img, dsize = (48,48), interpolation = cv2.INTER_CUBIC)
	return torch.Tensor(img.reshape(1,1,48,48))

model = Net()
model.load_state_dict(torch.load('model_12272018.pth'))

curr_emotion = "Neutral"
last_time = time.time()
while(True):
	screen = np.array(ImageGrab.grab(bbox = (300, 300, 800, 800)))
	processed_img = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
	new_screen = process_img(screen)

	x = new_screen
	x = image_to_tensor(x)
	output= model(x)
	_, pred = torch.max(output, 1)

	if (time.time() - last_time) > 1:
		print(Emotion[int(pred)])
		curr_emotion = Emotion[int(pred)]
		last_time = time.time()
	cv2.putText(processed_img, curr_emotion, (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
	cv2.imshow('window2', processed_img)
	#print('Frame rate of {:.3f}s.'.format(1/(time.time() - last_time)))
	# cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)) #Original Screen
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
