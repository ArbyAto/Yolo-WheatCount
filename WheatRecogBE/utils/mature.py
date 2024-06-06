import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def color_analysis(image):
    # Define color ranges for green, yellow, and brown
    green_lower = np.array([30, 100, 30])
    green_upper = np.array([85, 255, 85])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    brown_lower = np.array([10, 60, 20])
    brown_upper = np.array([20, 255, 200])

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    brown_mask = cv2.inRange(hsv_image, brown_lower, brown_upper)

    green_percentage = np.sum(green_mask) / green_mask.size
    yellow_percentage = np.sum(yellow_mask) / yellow_mask.size
    brown_percentage = np.sum(brown_mask) / brown_mask.size

    return green_percentage, yellow_percentage, brown_percentage

def classify_maturity(file_path):
    image = load_image(file_path)
    green_percentage, yellow_percentage, brown_percentage = color_analysis(image)
    # Adjust classification criteria based on more features
    # Normalize percentages
    total_percentage = green_percentage + yellow_percentage + brown_percentage
    green_percentage /= total_percentage
    yellow_percentage /= total_percentage
    brown_percentage /= total_percentage

    if green_percentage > 0.3:
        return 0 # "乳熟期"
    elif yellow_percentage > 0.1 and brown_percentage < 0.05:
        return 1 # "蜡熟期"
    elif brown_percentage > 0.05:
        return 2 # "完熟期"
    else:
        return 3 # "中间阶段"