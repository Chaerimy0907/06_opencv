# calcOpticalFlowPyrLK 추적 (수정된 버전)

import numpy as np, cv2

cap = cv2.VideoCapture('../img/walking.avi')
fps = cap.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(1000/fps)

# 추적 경로를 그리기 위한 랜덤 색상
color = np.random.randint(0, 255, (200, 3))
lines = None  # 추적 선을 그릴 이미지 저장 변수
prevImg = None  # 이전 프레임 저장 변수

# calcOpticalFlowPyrLK 중지 요건 설정
termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

print("옵티컬 플로우 추적 시작")
print("ESC: 종료, Backspace: 추적 이력 초기화")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # 비디오 끝에 도달하면 처음부터 다시 재생
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prevImg = None  # 초기화
        continue
        
    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 최초 프레임인 경우
    if prevImg is None:
        prevImg = gray
        # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
        lines = np.zeros_like(frame)
        # 추적 시작을 위한 코너 검출
        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
        print(f"초기 추적점 개수: {len(prevPt) if prevPt is not None else 0}")
    else:
        nextImg = gray
        
        # 추적할 점이 있는지 확인
        if prevPt is not None and len(prevPt) > 0:
            # 옵티컬 플로우로 다음 프레임의 코너점 찾기
            nextPt, status, err = cv2.calcOpticalFlowPyrLK(
                prevImg, nextImg, prevPt, None, criteria=termcriteria
            )
            
            # 유효한 추적점만 선별
            if nextPt is not None:
                # 대응점이 있는 코너, 움직인 코너 선별
                good_old = prevPt[status == 1]
                good_new = nextPt[status == 1]
                
                # 추적 경로 그리기
                for i, (old, new) in enumerate(zip(good_old, good_new)):
                    # 좌표를 정수로 변환 (중요!)
                    px, py = int(old.ravel()[0]), int(old.ravel()[1])
                    nx, ny = int(new.ravel()[0]), int(new.ravel()[1])
                    
                    # 유효한 좌표인지 확인
                    if (0 <= px < frame.shape[1] and 0 <= py < frame.shape[0] and
                        0 <= nx < frame.shape[1] and 0 <= ny < frame.shape[0]):
                        
                        # 이전 코너와 새로운 코너에 선 그리기
                        cv2.line(lines, (px, py), (nx, ny), color[i % len(color)].tolist(), 2)
                        # 새로운 코너에 점 그리기
                        cv2.circle(img_draw, (nx, ny), 2, color[i % len(color)].tolist(), -1)
                
                # 다음 프레임을 위한 데이터 업데이트
                prevImg = nextImg
                prevPt = good_new.reshape(-1, 1, 2)
                
                # 추적점이 너무 적어지면 새로 검출
                if len(prevPt) < 50:
                    print(f"추적점 부족 ({len(prevPt)}개), 새로운 특징점 검출")
                    new_points = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
                    if new_points is not None:
                        # 기존 점들과 새 점들을 합치기
                        if len(prevPt) > 0:
                            prevPt = np.vstack((prevPt, new_points))
                        else:
                            prevPt = new_points
                        
                        # 색상 배열도 확장
                        additional_colors = np.random.randint(0, 255, (len(new_points), 3))
                        color = np.vstack((color[:len(prevPt)-len(new_points)], additional_colors))
            else:
                print("옵티컬 플로우 실패, 새로운 특징점 검출")
                prevImg = None  # 초기화하여 새로 시작
        else:
            print("추적할 점이 없음, 새로운 특징점 검출")
            prevImg = None  # 초기화하여 새로 시작
    
    # 누적된 추적 선을 출력 이미지에 합성
    if lines is not None:
        img_draw = cv2.add(img_draw, lines)
    
    # 현재 추적점 개수 표시
    if prevPt is not None:
        cv2.putText(img_draw, f'Tracking Points: {len(prevPt)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('OpticalFlow-LK', img_draw)
    
    key = cv2.waitKey(delay) & 0xFF
    if key == 27:  # ESC: 종료
        break
    elif key == 8:  # Backspace: 추적 이력 지우기
        print("추적 이력 초기화")
        prevImg = None
        lines = None

print("옵티컬 플로우 추적 종료")
cv2.destroyAllWindows()
cap.release()