import cv2, numpy as np, glob, os

# 설정값
ratio = 0.7     # 좋은 매칭 선별 비율
MIN_MATCH = 10  # 최소 매칭점 수
detector = cv2.ORB_create(1000) # 특징점 개수 제한

# Flann 매칭기 객체 생성
FLANN_INDEX_LSH = 6 # LSH 알고리즘
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,       # 해시 테이블 개수
                    key_size = 12,          # 해시 키 크기
                    multi_probe_level = 1)  # 검색 레벨
search_params = dict(checks=32)     # 검색 시 확인할 리프 노드 수
matcher = cv2.FlannBasedMatcher(index_params, search_params)

def resize_image(img, max_width=400):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        img = cv2.resize(img, (max_width, new_h))
    return img

def search_book(query_img):
    # 이미지 전처리
    gray1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    kp1, desc1 = detector.detectAndCompute(gray1, None)

    results = {}

    # 책 커버 폴더에서 모든 이미지 파일 찾기
    book_paths = glob.glob('../img/books')

    for book_path in book_paths:
        cover = cv2.imread(book_path)
        cover = resize_image(cover)
        cv2.imshow('Searching...', cover)   # 검색 중인 책 표지 표시
        cv2.waitKey(5)  # 짧은 대기로 화면 업데이트

        # 데이터베이스 이미지 전처리 및 특징점 검출
        gray2 = cv2.cvtCoor(cover, cv2.COLOR_BGR2GRAY)
        kp2, desc2 = detector.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None:
            continue

        # KNN 매칭 (k=2 : 가장 가까운 2개 매칭점 반환)
        matches = matcher.knnMatch(desc1, desc2, 2)

        # Lovwe's 비율 테스트로 좋은 매칭 선별
        good_matches = [m[0] for m in matches if len(m)==2 and m[0].distance < m[1].distance * ratio]

        if len(good_matches) > MIN_MATCH:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                accuracy = float(mask.sum()) / mask.size
                results[book_path] = accuracy
    
    if len(results) > 0:
        results = sorted([(v, k) for (k, v) in results.items()], reverse=True)
    return results