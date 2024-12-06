from time import time, sleep

import cv2


def display():
    cv2.namedWindow('Capture Card')
    sleep(2)
    
    for capture_index in range(10):
        cap = cv2.VideoCapture(capture_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
        if not cap.isOpened():
            print(f"Cannot open capture {capture_index}")
            continue
    
        print(f'Displaying capture {capture_index}')
        
        while True:
            t = time()
            ret, frame = cap.read()
    
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
    
            cv2.imshow('Capture Card', frame)
            print(f"Capture {capture_index}")
    
            if cv2.waitKey(16) == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    display()
