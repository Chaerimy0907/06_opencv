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