import cv2
import numpy as np

def defect_area_cut_mix(normal_image, defect_image, defect_mask):
    # 결함 영역 추출
    defect_area = cv2.bitwise_and(defect_image, defect_image, mask=defect_mask)
    
    # 정상 이미지에서 결함이 들어갈 영역 선택
    x = np.random.randint(0, normal_image.shape[1] - defect_mask.shape[1])
    y = np.random.randint(0, normal_image.shape[0] - defect_mask.shape[0])
    
    # 결함 영역 합성
    roi = normal_image[y:y+defect_mask.shape[0], x:x+defect_mask.shape[1]]
    result = cv2.seamlessClone(defect_area, roi, defect_mask, (defect_mask.shape[1]//2, defect_mask.shape[0]//2), cv2.NORMAL_CLONE)
    
    # 합성 결과를 원본 이미지에 적용
    normal_image[y:y+defect_mask.shape[0], x:x+defect_mask.shape[1]] = result
    
    return normal_image

# 사용 예시
normal_img = cv2.imread('normal_image.jpg')
defect_img = cv2.imread('defect_image.jpg')
defect_mask = cv2.imread('defect_mask.png', 0)  # 그레이스케일로 읽기

augmented_img = defect_area_cut_mix(normal_img, defect_img, defect_mask)
cv2.imwrite('augmented_image.jpg', augmented_img)