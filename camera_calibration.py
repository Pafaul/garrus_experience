import configparser
import numpy as np
import cv2
import glob
import types
import os

video_capture_stream = None

def config_camera_with_checkers(get_img: types.GeneratorType,
                                chessboard_size: tuple,
                                dev: bool = False):

   
    squares = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    squares[:,:2] = np.mgrid[0:chessboard_size[1], 0:chessboard_size[0]].T.reshape(-1, 2)
    obj_points = []
    img_points = []

    board_founded = False

    cv2.namedWindow('camera')
    while not board_founded:
        img = next(get_img)

        cv2.imshow('camera', img)
        cv2.waitKey(1)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        board_founded, corners = cv2.findChessboardCorners(gray_img, chessboard_size, None)

        if board_founded:
            print('board founded')
            obj_points.append(squares)
            img_points.append(corners)

            if (dev):
                img_tmp = img.copy()
                cv2.drawChessboardCorners(img_tmp, chessboard_size, corners, board_founded)
                cv2.imshow('img', img_tmp)
                cv2.waitKey()

            ret, cam_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, 
                                                              img_points, 
                                                              gray_img.shape[::-1], 
                                                              None, None)
            break

        else:
            pass
    
    return [cam_matrix, dist]


def undistort_image(img: np.array,
                    cam_matrix,
                    dist,
                    dev=False):

    height, width = img.shape[:2]
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, 
                                                    dist, 
                                                    (width, height), 
                                                    1, 
                                                    (width, height))
    dst_img = cv2.undistort(img, cam_matrix, dist, None, new_cam_matrix)
    if dev:
        cv2.imshow('undistorted img', dst_img)
        cv2.waitKey(0)
        cv2.imwrite('undistorted.png', dst_img)

    print(roi)
    x, y, w, h = roi
    undistorted_img = dst_img[y:y+h, x:x+w]
    if dev:
        cv2.imshow('undistorted img', undistorted_img)
        cv2.waitKey(0)
        cv2.imwrite('undistorted_2.png', undistorted_img)
    
    return undistorted_img


def get_image_from_camera():
    global video_capture_stream

    if not video_capture_stream:
        video_capture_stream = cv2.VideoCapture(0)        
    
    else:
        while True:
            ret = False
            while not ret:
                ret, img = video_capture_stream.read()
            yield img


def get_image_for_tests():
    images = glob.glob(os.path.join('images', 'for_tests', '*.jpg'))

    for image in images:
        img = cv2.imread(image)
        if not img is None:
            yield img
        else:
            print('Cannot load image ' + image)


def main():
    global video_capture_stream

    get_img = get_image_from_camera()
    video_capture_stream = cv2.VideoCapture(0)

    parameters = config_camera_with_checkers(get_img,
                                             (5, 3),
                                             True)

    img = next(get_img)
    clear_image = undistort_image(img,
                                  parameters[0],
                                  parameters[1],
                                  True)

    cv2.imshow('cleared image', clear_image)
    pass


if __name__ == '__main__':
    main()
