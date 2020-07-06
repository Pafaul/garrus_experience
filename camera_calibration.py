import configparser
import numpy as np
import cv2
import glob
import types
import os

video_capture_stream = None
test_is_going_on = False

def create_confparser():
    config_parser = configparser.ConfigParser()
    config_parser.read(glob.glob('configuration.conf')[0])
    return config_parser

def parse_config():
    config_parser=create_confparser()
    use_camera = config_parser['camera_config']['use_camera'].strip() == 'True'
    if ( use_camera ):
        camera_id = int(config_parser['camera_config']['camera_id'])
    else:
        camera_id = -1

    camera_url = config_parser['camera_config']['camera_url']

    chessboard_size = tuple([ int(x.strip()) for x in config_parser['camera_config']['chessboard_size'].split(',') if x != ''])

    return [use_camera, camera_url, camera_id, chessboard_size]


def get_img_generator():
    use_camera, camera_url, camera_id, chessboard_size = parse_config()

    get_img = None
    if use_camera:
        global video_capture_stream

        if camera_url != '':
            if camera_url.isdigit():
                camera_url = int(camera_url)
            video_capture_stream = cv2.VideoCapture(camera_url)
        else:
            video_capture_stream = cv2.VideoCapture(0)
        
        if not video_capture_stream.isOpened():
            print('Cannot open camera')
            exit(1)
        else:
            video_capture_stream.read()
            get_img = get_image_from_camera()

    else:
        get_img = get_image_for_tests()

    return [ get_img, use_camera, chessboard_size ]


def get_image_from_camera():
    global video_capture_stream

    while test_is_going_on:
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
            yield None

    while True:
        print('No more images available')
        yield None


def resize_img(source_img: np.array):
    shape_of_src = source_img.shape[:2]
    if max(shape_of_src) > 1000:
        resize_coefficient = 1000/max(shape_of_src)
        result_size = tuple([int(x*resize_coefficient) for x in reversed(shape_of_src)])
        result_img = cv2.resize(source_img, 
                                result_size,
                                interpolation=cv2.INTER_CUBIC)
        return result_img
    return source_img


def config_camera_with_checkers(get_img: types.GeneratorType,
                                chessboard_size: tuple,
                                real_time: bool,
                                dev: bool = False):
   
    squares = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    squares[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_points = []
    img_points = []

    board_founded = False

    while not board_founded:
        img = next(get_img)

        if not img is None:
            img = resize_img(img)
            cv2.imshow('camera', img)
            if real_time:
                cv2.waitKey(1)
            else:
                cv2.waitKey(0)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            board_founded, corners = cv2.findChessboardCorners(gray_img, chessboard_size, None)

            if board_founded:
                print('board founded')
                obj_points.append(squares)
                corners_subpix = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_subpix)

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
        
        else: 
            return [None, None, None]

    return [img, cam_matrix, dist]


def undistort_image(img: np.array,
                    cam_matrix: np.array,
                    dist: np.array,
                    dev=False):

    height, width = img.shape[:2]
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, 
                                                    dist, 
                                                    (width, height), 
                                                    1, 
                                                    (width, height))
    
    print("New camera matrix: " + str(new_cam_matrix))
    dst_img = cv2.undistort(img, cam_matrix, dist, None, new_cam_matrix)
    if dev:
        cv2.imshow('undistorted img', dst_img)
        cv2.waitKey(0)
        cv2.imwrite('undistorted.png', dst_img)

    undistorted_img = None
    x, y, w, h = roi
    if w > 0 and h > 0:
        undistorted_img = dst_img[y:y+h, x:x+w]
    
        # if dev:
        #     cv2.imshow('undistorted img', undistorted_img)
        #     cv2.waitKey(0)
        #     cv2.imwrite('undistorted_2.png', undistorted_img)
    else:
        print('Calibration failed')
    
    return undistorted_img


def main():
    global test_is_going_on
    test_is_going_on = True

    get_img, use_camera, chessboard_size = get_img_generator()

    clear_image = None
    while clear_image is None:
        img, cam_matrix, dist = config_camera_with_checkers(get_img,
                                                            chessboard_size,
                                                            use_camera,
                                                            True)

        if img is not None:
            clear_image = undistort_image(img,
                                          cam_matrix,
                                          dist,
                                          True)
        else:
            clear_image = None
        
        if clear_image is None:
            print('Calibration Failed. Retry? [Y/n] ', end = '')
            retry = input()
            if retry.lower() == 'n':
                print('Result will not be recorded. Exiting.')
                break
            else:
                clear_image = None
        
        else:
            cv2.imshow('cleared image', clear_image)
            cv2.waitKey(0)
            print('Вас удовлетворяет результат?')
            retry = input()
            if retry.lower() == 'n':
                clear_image = None

    test_is_going_on = False
    if use_camera:
        video_capture_stream.release()


if __name__ == '__main__':
    main()
