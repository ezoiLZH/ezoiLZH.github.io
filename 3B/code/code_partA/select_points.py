import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def read(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    return img
def select_points(im1, im2, n_points=8):
    
    plt.figure(figsize=(20, 16))
    plt.title('Select Corresponding Points')
    plt.imshow(im1)
    points1 = plt.ginput(n_points, timeout=0)
    print("Selected points in image 1:", points1)
    plt.close()
    plt.imshow(im2)
    points2 = plt.ginput(n_points, timeout=0)
    print("Selected points in image 2:", points2)
    plt.close()
    return np.array(points1), np.array(points2)

def show(im1, im2, im1_pts, im2_pts):
    for pt in im1_pts:
        cv2.circle(im1, (int(pt[0]), int(pt[1])), 5, (255,0,0), -1)
    for pt in im2_pts:
        cv2.circle(im2, (int(pt[0]), int(pt[1])), 5, (255,0,0), -1)
    cv2.imshow('im1', im1)
    cv2.imshow('im2', im2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

im1 = read('../img/3_1.jpg')
im2 = read('../img/3_2.jpg')
im1_pts, im2_pts = select_points(im1, im2)
os.makedirs('./points', exist_ok=True)
np.savez('./points/3_12.npz', im1_pts=im1_pts, im2_pts=im2_pts)

# im1_pts, im2_pts = np.load('./points/5_12.npz')['im1_pts'], np.load('./points/5_12.npz')['im2_pts']

# show(im1, im2, im1_pts, im2_pts)