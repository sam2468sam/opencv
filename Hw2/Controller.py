import sys
import cv2
import numpy as np
import os.path
import random
import tensorflow

from HW2 import Ui_Form
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

class Controller(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.background_substraction)
        self.pushButton_2.clicked.connect(self.preprocessing)
        self.pushButton_3.clicked.connect(self.video_tracking)
        self.pushButton_4.clicked.connect(self.perspective_transform)
        self.pushButton_5.clicked.connect(self.image_reconstruction)
        self.pushButton_6.clicked.connect(self.compute_the_reconstruction_error)
        self.pushButton_9.clicked.connect(self.show_tensorboard)
        self.pushButton_10.clicked.connect(self.test)
        self.pushButton_11.clicked.connect(self.random_erasing)
        self.show()
        self.position = 0
        self.img_list = []
        self.reconstruction_error_list = []
         
    @pyqtSlot()
    def background_substraction(self):
        video = cv2.VideoCapture('./Q1_Image/bgSub.mp4')
        counter = 0
        frame_list = []
        frame_mean = 0
        frame_std = 0
        while(video.isOpened()):
            ret, frame = video.read()

            if ret == True :
                if(counter < 50):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('img1', gray)
                    cv2.waitKey(10)
                    gray = gray.astype(np.float32)
                    frame_list.append(gray)
                    counter = counter + 1
                    if(counter == 50):
                        for i in range(50):
                            frame_mean = frame_mean + frame_list[i]
                        frame_mean = frame_mean / 50
                        frame_msan = frame_mean.astype(np.uint8)
                        for i in range(50):
                            frame_list[i] = frame_list[i] - frame_mean
                            frame_list[i] = np.power(frame_list[i], 2)
                            frame_std = frame_std + frame_list[i]
                        frame_std = np.power(frame_std, 1/2)
                        frame_std = frame_std.astype(np.uint8)
                        for i in range(320):
                            for j in range(176):
                                if(frame_std[j][i] < 5):
                                    frame_std[j][i] = 5
                else :
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('img1', gray)
                    # for i in range(320):
                    #     for j in range(176):
                    #         if(gray[j][i] - frame_mean[j][i] > 5 * frame_std[j][i]):
                    #             gray[j][i] = 255
                    #         else :
                    #             gray[j][i] = 0
                    gray = gray - frame_mean - 5 * frame_std
                    ret , thresh1 = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
                    cv2.imshow('img', thresh1)
                    cv2.waitKey(30)
            else :
                break

        video.release()
        cv2.destroyAllWindows()

    def preprocessing(self):
        video = cv2.VideoCapture('./Q2_Image/opticalFlow.mp4')

        ret, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Setup SimpleBlobDetector parameters. 
        params = cv2.SimpleBlobDetector_Params() 

        # Change thresholds 
        params.minThreshold = 80
        params.maxThreshold = 300
        # params.minThreshold = 100
        # params.maxThreshold = 115

        # Filter by Area. 
        params.filterByArea = True 
        params.minArea = 35
        params.maxArea = 55
        # params.minArea = 20
        # params.maxArea = 100

        # Filter by Circularity 
        params.filterByCircularity = True 
        params.minCircularity = 0.8
        # params.minCircularity = 0.5

        # Filter by Convexity 
        params.filterByConvexity = True 
        params.minConvexity = 0.1
        # params.minConvexity = 0.8

        # Filter by Inertia 
        params.filterByInertia = True 
        params.minInertiaRatio = 0.4
        # params.minInertiaRatio = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)

        self.position = np.zeros((7, 2))

        for i in range(7):
            if(keypoints[i].pt[0] < 200):
                self.position[i][0] = keypoints[i].pt[0]
                self.position[i][1] = keypoints[i].pt[1]

        for i in range(7):
            img = cv2.line(frame, (int(self.position[i][0]) - 5, int(self.position[i][1])), (int(self.position[i][0]) + 5, int(self.position[i][1])), (0, 0, 255))
            img = cv2.line(frame, (int(self.position[i][0]), int(self.position[i][1]) - 5), (int(self.position[i][0]), int(self.position[i][1]) + 5), (0, 0, 255))
            img = cv2.rectangle(frame, (int(self.position[i][0] - 5), int(self.position[i][1]) - 5), (int(self.position[i][0]) + 5, int(self.position[i][1]) + 5), (0, 0, 255))

        # img1 = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('img', img)
        cv2.waitKey(0)

        video.release()
        cv2.destroyAllWindows()

    def video_tracking(self):
        video = cv2.VideoCapture('./Q2_Image/opticalFlow.mp4')

        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners = 7, qualityLevel = 0.1, minDistance = 1, blockSize = 1)

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15), maxLevel = 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Take first frame and find corners in it
        ret, old_frame = video.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        for i in range(7):
            p0[i][0][0] = self.position[i][0]
            p0[i][0][1] = self.position[i][1]
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        while(video.isOpened()):
            ret, frame = video.read()
            if ret == True :
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]

                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), (0, 0, 255), 2)
                    frame = cv2.circle(frame,(a,b),5,(0, 0, 255),-1)
                
                img = cv2.add(frame,mask)

                cv2.imshow('frame',img)
                cv2.waitKey(10)

                # Now update the previous frame and previous points
                old_gray = gray.copy()
                p0 = good_new.reshape(-1,1,2)
            else :
                break

        video.release()
        cv2.destroyAllWindows()
        
    def perspective_transform(self):
        # image or video or other
        detectType = 'video'
        detectPath = './Q3_Image/test4perspective.mp4'
        im_src = cv2.imread("./Q3_Image/rl.jpg")

        outputFile = "ar_out_py.avi"
        if (detectType is "image"):
            # Open the image file
            if not os.path.isfile(detectPath):
                print("Input image file ", detectPath, " doesn't exist")
                sys.exit(1)
            cap = cv2.VideoCapture(detectPath)
            outputFile = detectPath[:-4]+'_ar_out_py.jpg'
        elif (detectType is "video"):
            # Open the video file
            if not os.path.isfile(detectPath):
                print("Input video file ", detectPath, " doesn't exist")
                sys.exit(1)
            cap = cv2.VideoCapture(detectPath)
            outputFile = detectPath[:-4]+'_ar_out_py.avi'
            print("Storing it as :", outputFile)
        else:
            # Webcam input
            cap = cv2.VideoCapture(0)

        # Get the video writer initialized to save the output video
        if (detectType is not "image"):
            vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, (round(2*cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        winName = "Augmented Reality using Aruco markers in OpenCV"

        while cv2.waitKey(1) < 0:
            try:
                # get frame from the video
                hasFrame, frame = cap.read()

                # Stop the program if reached end of video
                if not hasFrame:
                    print("Done processing !!!")
                    print("Output file is stored as ", outputFile)
                    cv2.waitKey(3000)
                    break

                # Load the dictionary that was used to generate the markers.
                dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

                # Initialize the detector parameters using default values
                parameters = cv2.aruco.DetectorParameters_create()

                # Detect the markers in the image
                markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

                index = np.squeeze(np.where(markerIds == 25))
                refPt1 = np.squeeze(markerCorners[index[0]])[1]

                index = np.squeeze(np.where(markerIds == 33))
                refPt2 = np.squeeze(markerCorners[index[0]])[2]

                distance = np.linalg.norm(refPt1-refPt2)

                scalingFac = 0.02
                pts_dst = [[refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]]
                pts_dst = pts_dst + [[refPt2[0] + round(scalingFac*distance), refPt2[1] - round(scalingFac*distance)]]

                index = np.squeeze(np.where(markerIds == 30))
                refPt3 = np.squeeze(markerCorners[index[0]])[0]
                pts_dst = pts_dst + [[refPt3[0] + round(scalingFac*distance), refPt3[1] + round(scalingFac*distance)]]

                index = np.squeeze(np.where(markerIds == 23))
                refPt4 = np.squeeze(markerCorners[index[0]])[0]
                pts_dst = pts_dst + [[refPt4[0] - round(scalingFac*distance), refPt4[1] + round(scalingFac*distance)]]

                pts_src = [[0, 0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]]

                pts_src_m = np.asarray(pts_src)
                pts_dst_m = np.asarray(pts_dst)

                # Calculate Homography
                h, status = cv2.findHomography(pts_src_m, pts_dst_m)

                # Warp source image to destination based on homography
                warped_image = cv2.warpPerspective(im_src, h, (frame.shape[1], frame.shape[0]))

                # Prepare a mask representing region to copy from the warped image into the original frame.
                mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA)

                # Erode the mask to not copy the boundary effects from the warping
                element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.erode(mask, element, iterations=3)

                # Copy the mask into 3 channels.
                warped_image = warped_image.astype(float)
                mask3 = np.zeros_like(warped_image)
                for i in range(0, 3):
                    mask3[:, :, i] = mask/255

                # Copy the warped image into the original frame in the mask region.
                warped_image_masked = cv2.multiply(warped_image, mask3)
                frame_masked = cv2.multiply(frame.astype(float), 1-mask3)
                im_out = cv2.add(warped_image_masked, frame_masked)

                # Showing the original image and the new output image side by side
                concatenatedOutput = cv2.hconcat([frame.astype(float), im_out])
                cv2.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))

                # Write the frame with the detection boxes
                if (detectType is "image"):
                    cv2.imwrite(outputFile, concatenatedOutput.astype(np.uint8))
                else:
                    vid_writer.write(concatenatedOutput.astype(np.uint8))

            except Exception as inst:
                print(inst)

        cv2.destroyAllWindows()
        if 'vid_writer' in locals():
            vid_writer.release()
            print('Video writer released..')

    def image_reconstruction(self):
        pca = PCA(n_components=10)

        for i in range(1,35):
            img = cv2.imread('./Q4_Image/%d.jpg' %i)

            test = np.reshape(img, (100, 300))

            transformed_img = pca.fit_transform(test)

            reconstructed_img = pca.inverse_transform(transformed_img)
            reconstructed_img = cv2.convertScaleAbs(reconstructed_img)
            reconstructed_img = np.reshape(reconstructed_img, (100, 100, 3))

            self.img_list.append(img)
            self.img_list.append(reconstructed_img)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            reconstructed_gray = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2GRAY)

            reconstruction_error = 0

            for i in range(100):
                for j in range(100):
                    reconstruction_error = reconstruction_error + abs(gray[i][j] - reconstructed_gray[i][j])

            self.reconstruction_error_list.append(reconstruction_error)

        img1 = np.hstack((self.img_list[0],self.img_list[2],self.img_list[4],self.img_list[6],self.img_list[8],self.img_list[10],self.img_list[12],self.img_list[14],self.img_list[16],self.img_list[18],self.img_list[20],self.img_list[22],self.img_list[24],self.img_list[26],self.img_list[28],self.img_list[30],self.img_list[32]))
        img2 = np.hstack((self.img_list[1],self.img_list[3],self.img_list[5],self.img_list[7],self.img_list[9],self.img_list[11],self.img_list[13],self.img_list[15],self.img_list[17],self.img_list[19],self.img_list[21],self.img_list[23],self.img_list[25],self.img_list[27],self.img_list[29],self.img_list[31],self.img_list[33]))
        img3 = np.hstack((self.img_list[34],self.img_list[36],self.img_list[38],self.img_list[40],self.img_list[42],self.img_list[44],self.img_list[46],self.img_list[48],self.img_list[50],self.img_list[52],self.img_list[54],self.img_list[56],self.img_list[58],self.img_list[60],self.img_list[62],self.img_list[64],self.img_list[66]))
        img4 = np.hstack((self.img_list[35],self.img_list[37],self.img_list[39],self.img_list[41],self.img_list[43],self.img_list[45],self.img_list[47],self.img_list[49],self.img_list[51],self.img_list[53],self.img_list[55],self.img_list[57],self.img_list[59],self.img_list[61],self.img_list[63],self.img_list[65],self.img_list[67]))
        img = np.vstack((img1, img2, img3, img4))

        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compute_the_reconstruction_error(self):
        print(self.reconstruction_error_list)

    def show_tensorboard(self):
        accuracy = cv2.imread('./Q5_Image/epoch_accuracy.png')
        loss = cv2.imread('./Q5_Image/epoch_loss.png')
        result = cv2.imread('./Q5_Image/result.png')

        cv2.namedWindow('accuracy', cv2.WINDOW_NORMAL)
        cv2.namedWindow('loss', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('accuracy',accuracy)
        cv2.imshow('loss',loss)
        cv2.imshow('result',result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def test(self):
        model = tensorflow.keras.models.load_model("my_model.h5")

        # for i in range(100):

        index = random.randint(1,9999)

        img = cv2.imread('./Q5_Image/test1/%d.jpg' %index)

        new_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

        new_img1 = np.reshape(new_img, (1, 224, 224, 3))

        ans = model.predict(new_img1)

        img_block = np.zeros((20, 224, 3))
        new_img[185:205, 0:224] = img_block

        if(ans[0][0] > ans[0][1]):
            cv2.putText(new_img, 'Class : cat', (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else :
            cv2.putText(new_img, 'Class : dog', (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('result',new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def random_erasing(self):
        label = ["Before Random-Erasing","After Random-Erasing"]

        result = cv2.imread('./Q5_Image/result.png')
        result_erasing = cv2.imread('./Q5_Image/result1.png')

        # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('result_erasing', cv2.WINDOW_NORMAL)

        cv2.imshow('result',result)
        cv2.imshow('result_erasing',result_erasing)

        ans = [0.94,0.98]

        fig = plt.figure()
        plt.xlabel("methods")
        plt.ylabel("accuracy")
        plt.xticks(range(10))
        plt.bar(label,ans)
        plt.show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
     app = QApplication(sys.argv)
     window = Controller()
     sys.exit(app.exec_())


