# 필요한 라이브러리 opencv-python
import cv2

# webcam 열기 - 일반적으로 카메라가 하나 달려 있으면 0을 쓰면 됨.
# 여러개의 카메라가 있는 경우는 번호를 확인하고 사용.
cap = cv2.VideoCapture(0)

while True:     # 무한 반복 (아래의 break에 의해서 종료됨.)
    ret, frame = cap.read()     # 한개의 화면을 읽어옴. ret는 화면이 잘 읽어졌는지 확인하는 것, frame은 화면 내용

    if ret:     # 화면이 잘 읽어졌다면...
        cv2.imshow("live feed", frame)

    key = cv2.waitKey(1)        # 키 입력을 기다림.
    if key == ord("q"):         # q 키가 눌러졌다면,
        break                   # while을 break 함.

cap.release()           # 카메라를 놔주고
cv2.destroyAllWindows() # 열린 화면을 닫아주고.. 그러면 끝.