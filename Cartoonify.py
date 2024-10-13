import cv2
import numpy as np
import os
from datetime import datetime

def cartoonize_image(image, k=6, scale=0.5):

    small_img = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    smooth_img = cv2.bilateralFilter(small_img, d=9, sigmaColor=75, sigmaSpace=75)

    data = np.float32(smooth_img).reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    stylized_img = centers[labels.flatten()].reshape(small_img.shape)

    edges = cv2.Canny(small_img, 100, 200)

    edges_inv = cv2.bitwise_not(edges)

    edges_inv_bgr = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    cartoon_img = cv2.bitwise_and(stylized_img, edges_inv_bgr)

    cartoon_img_upscaled = cv2.resize(cartoon_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return cartoon_img_upscaled

def save_images(original, cartoonized):
    """Saves both the original and cartoonized frames to the respective folders."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists('real'):
        os.makedirs('real')
    if not os.path.exists('cartoon'):
        os.makedirs('cartoon')

    real_path = os.path.join('real', f"real_{timestamp}.png")
    cartoon_path = os.path.join('cartoon', f"cartoon_{timestamp}.png")

    cv2.imwrite(real_path, original)
    cv2.imwrite(cartoon_path, cartoonized)

cap = cv2.VideoCapture(1)  # (0 or 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


cv2.namedWindow("Real-Time Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cartoonized Live Feed", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    cartoon_frame = cartoonize_image(frame, k=6, scale=0.5)  

    cv2.imshow('Real-Time Video', frame)
    cv2.imshow('Cartoonized Live Feed', cartoon_frame)
    
    key = cv2.waitKey(1)  # Change to 1 for smooth video feed
    
    if key == ord('e') or key == ord('E'):  # Press 'E' to save both real and cartoonized images
        save_images(frame, cartoon_frame)
        print("Images saved!")
    
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
















