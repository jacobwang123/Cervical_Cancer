#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Jun 03

@author: jacob
"""

# This kernel aims at segmenting the cervix using the technique presented in this paper:
# https://www.researchgate.net/publication/24041301_Automatic_Detection_of_Anatomical_Landmarks_in_Uterine_Cervix_Images

import numpy as np
import cv2
import math
from glob import glob
import os
from multiprocessing import Pool, cpu_count

USER_ROOT = os.path.dirname(os.path.realpath(__file__))
TRAIN_DATA = "train"
TEST_DATA = "test"
ADDITIONAL_DATA = "additional"

class Segmenter(object):
	def __init__(self, img_path):
		self.img_path = img_path

	def get_image_data(self):
		"""
		Method to get image data as np.array specifying image id and type
		"""
		img = cv2.imread(self.img_path)
		assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	# First, we crop the image in order to remove the circular frames that might be present.
	# This is done by finding the largest inscribed rectangle to the thresholded image.
	# The image is then cropped to this rectangle. (see these videos for an explanation of the algorithm:
	# https://www.youtube.com/watch?v=g8bSdXCG-lA, https://www.youtube.com/watch?v=VNbkzsnllsU)

	def maxHist(self, hist):
		maxArea = (0, 0, 0)
		height = []
		position = []
		for i in range(len(hist)):
			if (len(height) == 0):
				if (hist[i] > 0):
					height.append(hist[i])
					position.append(i)
			else: 
				if (hist[i] > height[-1]):
					height.append(hist[i])
					position.append(i)
				elif (hist[i] < height[-1]):
					while (height[-1] > hist[i]):
						maxHeight = height.pop()
						area = maxHeight * (i-position[-1])
						if (area > maxArea[0]):
							maxArea = (area, position[-1], i)
						last_position = position.pop()
						if (len(height) == 0):
							break
					position.append(last_position)
					if (len(height) == 0):
						height.append(hist[i])
					elif(height[-1] < hist[i]):
						height.append(hist[i])
					else:
						position.pop()    
		while (len(height) > 0):
			maxHeight = height.pop()
			last_position = position.pop()
			area =  maxHeight * (len(hist) - last_position)
			if (area > maxArea[0]):
				maxArea = (area, len(hist), last_position)
		return maxArea

	def maxRect(self, img):
		maxArea = (0, 0, 0)
		addMat = np.zeros(img.shape)
		for r in range(img.shape[0]):
			if r == 0:
				addMat[r] = img[r]
				area = self.maxHist(addMat[r])
				if area[0] > maxArea[0]:
					maxArea = area + (r,)
			else:
				addMat[r] = img[r] + addMat[r-1]
				addMat[r][img[r] == 0] *= 0
				area = self.maxHist(addMat[r])
				if area[0] > maxArea[0]:
					maxArea = area + (r,)
		return (int(maxArea[3]+1-maxArea[0]/abs(maxArea[1]-maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])

	def cropCircle(self):
		img = self.get_image_data()

		if(img.shape[0] > img.shape[1]):
			tile_size = (int(img.shape[1]*256/img.shape[0]),256)
		else:
			tile_size = (256, int(img.shape[0]*256/img.shape[1]))

		img = cv2.resize(img, dsize=tile_size)

		boundaries = ([50, 0, 0], [255, 200, 255])
		lower = np.array(boundaries[0], dtype = "uint8")
		upper = np.array(boundaries[1], dtype = "uint8")
		mask = cv2.inRange(img, lower, upper)
		img_pink = cv2.bitwise_and(img, img, mask = mask)

		gray = cv2.cvtColor(img_pink, cv2.COLOR_RGB2GRAY)
		if cv2.countNonZero(gray) < gray.shape[0] * gray.shape[1] * 0.3:
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		_, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

		_, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

		main_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]
				
		ff = np.zeros((gray.shape[0],gray.shape[1]), 'uint8') 
		cv2.drawContours(ff, main_contour, -1, 1, 15)
		ff_mask = np.zeros((gray.shape[0]+2,gray.shape[1]+2), 'uint8')
		cv2.floodFill(ff, ff_mask, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 1)
		
		rect = self.maxRect(ff)
		rectangle = [min(rect[0],rect[2]), max(rect[0],rect[2]), min(rect[1],rect[3]), max(rect[1],rect[3])]
		img_crop = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
		img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
		return img_crop

def processingImage(which, type):
	type_i_files = glob(os.path.join(USER_ROOT, which, type, "*.jpg"))
	for img_path in type_i_files:
		s = Segmenter(img_path)
		img = s.cropCircle()
		cv2.imwrite(os.path.join(USER_ROOT, 'pre_processed', type, img_path.split('/')[-1]), img)
	return

if __name__ == '__main__':
	types = ['Type_1','Type_2','Type_3']
	atypes = ['AType_1','AType_2','AType_3']
	ntypes = ['NType_1','NType_2','NType_3']
	ttypes = ['test_stg2']
	# p = Pool(3)
	# p.map(processingImage, [ADDITIONAL_DATA]*3, types)
	for type in ttypes:
		processingImage(ADDITIONAL_DATA, type)
