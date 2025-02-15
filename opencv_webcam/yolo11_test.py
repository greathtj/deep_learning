import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")      # detection model
cap = cv2.VideoCapture(0)       # check your camera number

while True:
    ret, frame = cap.read()
    # print(frame)
    if ret:
        results = model(frame, verbose=False)   # inference
        frame = results[0].plot()
        detected_cls = results[0].boxes.cls.tolist()    # detected classes

        print("================")
        for ndx, cls in enumerate(detected_cls):        # for each classes detected
            msg = f"{results[0].names[cls]} "           # name of the object detected
            msg += f"at {results[0].boxes.xywhn[ndx][0].item():.2f}, {results[0].boxes.xywhn[ndx][1].item():.2f} "      # object position in the image
            msg += f"with conf {results[0].boxes.conf[ndx].item() * 100:.1f} %"     # how sure it is
            print(msg)
        
        cv2.imshow("webcam test", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()