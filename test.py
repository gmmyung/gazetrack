import cv2
from pynput import mouse, keyboard
import dlib
from pathlib import Path
import csv

def main():
    initialize()
    start()


def initialize():
    global capture
    global detector
    global predictor
    global cropsize
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cropsize = 100
    init_csv()

def init_csv():
    global counter
    global writer
    global csv_file
    my_file = Path("data/cords.csv")
    if not my_file.is_file():
        csv_file = open("data/cords.csv", 'w', newline='')
        counter = 0
        print("generating data.csv.....")
    else:
        with open("data/cords.csv", 'r') as csv_file:
            print("reading previous coordfiles.....")
            counter = 0
            csv_read = csv.reader(csv_file, delimiter=',')
            for row in csv_read:
                counter += 1
            csv_file = open("data/cords.csv", 'a', newline='')
    writer = csv.writer(csv_file)

def on_click(x, y, button, pressed):
    global counter 
    if pressed:
        ret, frame = capture.read()
        img = frame
        rects = detector(img, 1)
        for i, rect in enumerate(rects):
            l = rect.left()
            t = rect.top()
            b = rect.bottom()
            r = rect.right()
            shape = predictor(img, rect)
            disp = img
            cordlist = [counter]
            for j in range(68):
                p, q = shape.part(j).x, shape.part(j).y
                #cv2.circle(disp, (p, q), 1, (0, 0, 255), -1)
                cordlist.append(p)
                cordlist.append(q)
            cordlist.append(x)
            cordlist.append(y)
            #writer.writerow(cordlist)
            #cv2.rectangle(disp, (l, t), (r, b), (0, 255, 0), 2)
            cv2.imshow("CAM", disp)
            
            left_eye_x= sum([shape.part(j).x for j in [36, 37, 38, 39, 40, 41]]) // 6
            left_eye_y= sum([shape.part(j).y for j in [36, 37, 38, 39, 40, 41]]) // 6
            right_eye_x= sum([shape.part(j).x for j in [42, 43, 44, 45, 46, 47]]) // 6
            right_eye_y= sum([shape.part(j).y for j in [42, 43, 44, 45, 46, 47]]) // 6
            cropped_left = img[left_eye_y-cropsize//2: left_eye_y+cropsize//2, left_eye_x-cropsize//2: left_eye_x+cropsize//2]
            cropped_right = img[right_eye_y-cropsize//2: right_eye_y+cropsize//2, right_eye_x-cropsize//2: right_eye_x+cropsize//2]
            cv2.imshow("Right eye", cropped_right)
            cv2.imshow("Left eye", cropped_left)
            cropped_left = cv2.cvtColor(cropped_left, cv2.COLOR_BGR2GRAY)
            cropped_right = cv2.cvtColor(cropped_right, cv2.COLOR_BGR2GRAY)
            
            if cv2.imwrite('./data/left_eye/'+str(counter)+'.png',cropped_left) and cv2.imwrite('./data/right_eye/'+str(counter)+'.png',cropped_right):
                print(cordlist)
                writer.writerow(cordlist)
                csv_file.flush()
                print("coord saved")
                counter += 1
                print(counter)

def on_press(key):
    global break_program
    if key == keyboard.Key.esc:
        print ('end pressed')
        csv_file.close()
        return False


def start():
    listener_mouse = mouse.Listener(on_click=on_click)
    listener_mouse.start()
    print("Start logging!")
    with keyboard.Listener(on_press=on_press) as listener_keyboard:
        listener_keyboard.join()
    
main()