import cv2, numpy as np, glob, os, time, webbrowser

# 설정값
ratio = 0.7     # 좋은 매칭 선별 비율
MIN_MATCH = 10  # 최소 매칭점 수
detector = cv2.ORB_create(1000) # 특징점 개수 제한

book_links = {'book21.jpg': 'https://www.yes24.com/product/goods/117762049'}

# Flann 매칭기 객체 생성
FLANN_INDEX_LSH = 6 # LSH 알고리즘
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,       # 해시 테이블 개수
                    key_size = 12,          # 해시 키 크기
                    multi_probe_level = 1)  # 검색 레벨
search_params = dict(checks=32)     # 검색 시 확인할 리프 노드 수
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# 이미지 크기 변경 (성능 최적화)
def resize_image(img, max_width=400):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        img = cv2.resize(img, (max_width, new_h))
    return img

# 책 이미지 검색 함수
def search(img):
    # 이미지 전처리
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(gray1, None)

    results = {}

    # 책 커버 폴더에서 모든 이미지 파일 찾기
    book_paths = glob.glob('../img/books/*.*')

    for book_path in book_paths:
        cover = cv2.imread(book_path)
        cover = resize_image(cover)
        cv2.imshow('Searching...', cover)   # 검색 중인 책 표지 표시
        cv2.waitKey(5)  # 짧은 대기로 화면 업데이트

        # 데이터베이스 이미지 전처리 및 특징점 검출
        gray2 = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            continue

        # KNN 매칭 (k=2 : 가장 가까운 2개 매칭점 반환)
        matches = matcher.knnMatch(desc1, desc2, 2)

        # Lovwe's 비율 테스트로 좋은 매칭 선별
        good_matches = [m[0] for m in matches 
                        if len(m)==2 and m[0].distance < m[1].distance * ratio]

        if len(good_matches) > MIN_MATCH:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                accuracy = float(mask.sum()) / mask.size
                results[book_path] = accuracy
    
    if len(results) > 0:
        results = sorted([(v, k) for (k, v) in results.items()
                          if v > 0], reverse=True)
    return results

cap = cv2.VideoCapture(0)
win_name = "Capture ROI"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE로 ROI 선택
        x, y, w, h = cv2.selectROI(win_name, frame, False)
        if w and h:
            roi = frame[y:y+h, x:x+w]
            cv2.imshow("ROI", roi)
            cv2.imwrite("query.jpg", roi)  # 나중에 불러와서 매칭용으로 쓰기
            print("Saved as query.jpg")
            break

cap.release()
cv2.destroyAllWindows()

# 결과 처리 및 링크 열기
if roi is not None:
    results = search(roi)
    if len(results) == 0:
        print("No matched book cover found.")
    else:
        for i, (accuracy, cover_path) in enumerate(results):
            print(f"{i}: {cover_path} - 정확도: {accuracy:.2%}")

            if i == 0:
                cover = cv2.imread(cover_path)
                cv2.imshow('Result', cover)

                fname = os.path.basename(cover_path)
                if accuracy >= 0.90 and fname in book_links:
                    print(f">>> 정확도 {accuracy:.2%}로 링크 열기: {book_links[fname]}")
                    time.sleep(1)
                    webbrowser.open(book_links[fname])
                else:
                    print(">>> 정확도 부족 또는 등록된 링크 없음.")

cv2.waitKey()
cv2.destroyAllWindows()