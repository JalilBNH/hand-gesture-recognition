import numpy as np
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def skin_segmentation(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return mask

def find_hand_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour
    return None

def count_fingers(conv_def, depth_thresold=10000):
    count = 0
    
    for conv_def_idx in conv_def.squeeze():
        _, _, _, depth = conv_def_idx
        if depth > depth_thresold:
            count += 1
    
    print(f'nombre de doigts : {count}')
    return count

def plot_results(image, contour, conv_def, nb_doigts):
    contour_squeeze = contour.squeeze()
    convex_pts = cv2.convexHull(contour).squeeze()
    thresh = 10
    plt.subplot(121)
    plt.imshow(image[:,:, ::-1]), plt.title('images originale'),plt.axis('off')
    plt.subplot(122)
    plt.gca().invert_yaxis()
    plt.plot(convex_pts[:, 0], convex_pts[:, 1], color='red', label='enveloppe convexe')
    plt.scatter(contour_squeeze[:, 0], contour_squeeze[:, 1], s=3, label='contour de la main')
    for idx in conv_def.squeeze():    
        plt.scatter(contour_squeeze[idx[0] +thresh:idx[1] - thresh, 0], contour_squeeze[idx[0] + thresh:idx[1] - thresh, 1], color='orange', s=3)
    plt.scatter([], [], color='orange', s=3, label='défaut de convexité')
    plt.title(f'Nombre de doigts compté : {nb_doigts}')
    plt.axis('off')
    plt.legend()
    plt.show()



def main():
    img_path = './images_test/4.jpg'
    
    image = cv2.imread(img_path)
    seg_image = skin_segmentation(image)
    contour_image = find_hand_contours(seg_image)
    convex_hull = cv2.convexHull(contour_image, returnPoints=False)
    conv_def = cv2.convexityDefects(contour_image, convex_hull)
    nb_doigts = count_fingers(conv_def)
    plot_results(image, contour_image, conv_def, nb_doigts)
    
    
    




if __name__ == '__main__':
    main()