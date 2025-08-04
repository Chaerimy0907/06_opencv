'''
기존 방식 (ORB + FLANN 매칭) 
- 바코드는 특징점이 거의 없고, 패턴도 단순해서 ORB로 매칭 불가

수정 방식 cv2.matchTemplate() 방식 사용
- 바코드처럼 단순한 직선 패턴도 매칭 가능
- 특징점 필요 없음 -> detectAndCompute() 불필요
'''

import cv2, numpy as np

# 초기 설정
img1 = None # 참조 이미지 (ROI 선택 영역)
win_name = 'Camera Matching'
MIN_MATCH = 10  # 최소 매칭점 개수 (이 값 이하면 매칭 실패로 간주)

# ORB 검출기 생성
# ORB_create(1000) - 이미지에서 1000개의 특징점을 찾는 알고리즘
detector = cv2.ORB_create(1000)

# Flann 추출기 생성
# 두 이미지의 특징점을 빠르게 매칭
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 카메라 캡쳐 연결 및 프레임 크기 축소
cap = cv2.VideoCapture(0)              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():       
    ret, frame = cap.read() 
    if not ret:
        break
    
    if img1 is None:  # ROI 없으면 그대로 출력
        res = frame.copy()
    else:
        img2 = frame.copy()
        # [step 1]
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 참조 이미지
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 현재 카메라 영상
        
        # [step 2]
        # 키포인트와 디스크립터 추출
        # kp : keypoint 특징점의 위치 정보
        # desc : 특징점의 특성을 숫자로 표현
        kp1, desc1 = detector.detectAndCompute(gray1, None) # 참조 이미지의 특징점
        kp2, desc2 = detector.detectAndCompute(gray2, None)  # 카메라 이미지의 특징점

        # 디스크립터가 없으면 건너뛰기
        if desc1 is not None and desc2 is not None and len(desc1) >= 2 and len(desc2) >= 2:
            matches = matcher.knnMatch(desc1, desc2, 2)

            # [step 4]
            # 이웃 거리의 75%로 좋은 매칭점 추출
            ratio = 0.75
            good_matches = []
            for m, n in matches:
                if m.distance < n.distance * ratio: # 1등이 2등보다 25% 이상 좋으면
                    good_matches.append(m)

            print(f'good matches: {len(good_matches)}')
        
            # matchesMask 초기화를 None으로 설정
            matchesMask = None

            # 좋은 매칭점 최소 갯수 이상 인 경우
            if len(good_matches) >= MIN_MATCH: 
            
                # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
            
                # 원근 변환 행렬 구하기
                # RANSAC : 잘못된 매칭점 outline 제거
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
                if mtrx is not None and mask.sum() > MIN_MATCH:
                    # 결과 시각화
                    # 원본 영상 좌표로 원근 변환 후 영역 표시
                    h,w, = img1.shape[:2]
                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts,mtrx)
                    #img2 = cv2.polylines(img2,[np.int32(dst)],True, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(img2, "Matched", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    matchesMask = mask.ravel().tolist()
            
            # 매칭선 그리기
            res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                matchColor = (0, 255, 0),
                                matchesMask=matchesMask,
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        else:
            res = frame.copy()   
    
    # 결과 출력
    cv2.imshow(win_name, res)

    key = cv2.waitKey(1)
    if key == 27:    # Esc, 종료
        break          
    elif key == ord(' '): # 스페이스바를 누르면 ROI로 img1 설정
        x,y,w,h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
            print("ROI 선택됨 : (%d, %d, %d, %d)"%(x,y,w,h))
else:
    print("can't open camera.")

cap.release()                          
cv2.destroyAllWindows()
