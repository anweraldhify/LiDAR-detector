import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# From Github https://github.com/balcilar/DenseDepthMap
def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1
    
    mX = np.zeros((m,n)) + np.float("inf")
    mY = np.zeros((m,n)) + np.float("inf")
    mD = np.zeros((m,n))
    mX[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]),np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i,j] = mX[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmY[i,j] = mY[i : (m - ng + i), j : (n - ng + j)] - grid - 1 +i
            KmD[i,j] = mD[i : (m - ng + i), j : (n - ng + j)]
    S = np.zeros_like(KmD[0,0])
    Y = np.zeros_like(KmD[0,0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i,j] * KmX[i,j] + KmY[i,j] * KmY[i,j])
            Y = Y + s * KmD[i,j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m,n))
    out[grid + 1 : -grid, grid + 1 : -grid] = Y/S
    return out
    
# Class for the calibration matrices for KITTI data
class Calibration:
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])

        self.L2C = calibs['Tr_velo_to_cam']
        self.L2C = np.reshape(self.L2C, [3,4])

        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

    @staticmethod
    def read_calib_file(filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    # From LiDAR coordinate system to Camera Coordinate system
    def lidar2cam(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1))))
        pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(self.L2C))
        pts_3d_cam_rec = np.transpose(np.dot(self.R0, np.transpose(pts_3d_cam_ref)))
        return pts_3d_cam_rec
    
    # From Camera Coordinate system to Image frame
    def rect2Img(self, rect_pts):
        n = rect_pts.shape[0]
        points_hom = np.hstack((rect_pts, np.ones((n,1))))
        points_2d = np.dot(points_hom, np.transpose(self.P)) # nx3
        points_2d[:,0] /= points_2d[:,2]
        points_2d[:,1] /= points_2d[:,2]
        
        mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= 1242) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= 375)
        mask = mask & (rect_pts[:,2] > 2)
        return points_2d[mask,0:2], mask
        
# Change the root to where you saved the KITTI data
root = "/RVW/Sem8/Elser/KITTI/"
image_dir = os.path.join(root, "image_2")
velodyne_dir = os.path.join(root, "velodyne")
calib_dir = os.path.join(root, "calib")

imgs_in_file = os.listdir(image_dir)
imgs_in_file.sort()
print(imgs_in_file[0])
# Data id
cur_id = 0
# Loading the image

# imgs_in_file[1885:]
# imgs_in_file[6323:]

for i in imgs_in_file:
    img0 = cv2.imread(os.path.join(image_dir, "%06d.png" % cur_id))
    if img0.shape != (375, 1242, 3):
        img0 = cv2.resize(img0, dsize=(1242, 375))
    # Loading the LiDAR data
    lidar0  = np.fromfile(os.path.join(velodyne_dir, "%06d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
    # Loading Calibration
    calib = Calibration(os.path.join(calib_dir, "%06d.txt" % cur_id))
    # From LiDAR coordinate system to Camera Coordinate system
    lidar_rect = calib.lidar2cam(lidar0[:,0:3])
    # From Camera Coordinate system to Image frame
    lidarOnImage, mask = calib.rect2Img(lidar_rect)
    # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
    lidarOnImage = np.concatenate((lidarOnImage, lidar0[mask,3].reshape(-1,1)), 1)
    
    # Generate the LiDAR points on the image
    # plt.figure(figsize = (20, 10)) # Only to make the image size expand
    # plt.axis('off') # removes the axis
    # plt.imshow(img0)
    # plt.scatter(lidarOnImage[:,0], lidarOnImage[:,1], c = lidarOnImage[:,2], s = 5)
    # plt.savefig("intensity_on_image.png")
    
    # Make the intesity map
    out = dense_map(lidarOnImage.T, img0.shape[1],img0.shape[0], 4)
    plt.figure(figsize=(20,40))
    plt.imsave(i, out)
    cur_id = cur_id + 1