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

cap = cv2.VideoCapture(1)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Real-Time Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cartoonized Live Feed", cv2.WINDOW_NORMAL)

is_recording = False
real_writer = None
cartoon_writer = None

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    cartoon_frame = cartoonize_image(frame, k=6, scale=0.5)

    cv2.imshow('Real-Time Video', frame)
    cv2.imshow('Cartoonized Live Feed', cartoon_frame)
    
    key = cv2.waitKey(1)

    if key == ord('e') or key == ord('E'):  
        cv2.imwrite(f"real/real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", frame)
        cv2.imwrite(f"cartoon/cartoon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", cartoon_frame)
        print("Images saved!")

    if key == ord('v') or key == ord('V'):  # Start/Stop video recording
        if is_recording:
            # Stop recording
            is_recording = False
            real_writer.release()
            cartoon_writer.release()
            print("Recording stopped!")
        else:
            # Start recording
            is_recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if not os.path.exists('real-video'):
                os.makedirs('real-video')
            if not os.path.exists('cartoon-video'):
                os.makedirs('cartoon-video')

            real_writer = cv2.VideoWriter(f"real-video/real_{timestamp}.avi",
                                          cv2.VideoWriter_fourcc(*'XVID'),
                                          20.0, (frame.shape[1], frame.shape[0]))

            cartoon_writer = cv2.VideoWriter(f"cartoon-video/cartoon_{timestamp}.avi",
                                             cv2.VideoWriter_fourcc(*'XVID'),
                                             20.0, (cartoon_frame.shape[1], cartoon_frame.shape[0]))

            print("Recording started!")

    if is_recording:
        real_writer.write(frame)
        cartoon_writer.write(cartoon_frame)
    
    if key == 27:
        break

cap.release()
if is_recording:
    real_writer.release()
    cartoon_writer.release()
cv2.destroyAllWindows()

















