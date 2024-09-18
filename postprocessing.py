import cv2
import numpy as np
from skimage import measure

def post_process_segmentation(segmentation_mask, min_area=100, min_score=0.5):
    # 레이블링
    labeled_mask = measure.label(segmentation_mask > min_score)
    
    # 각 연결 요소의 속성 분석
    regions = measure.regionprops(labeled_mask)
    
    # 결함 목록 초기화
    defects = []
    
    for region in regions:
        # 면적이 너무 작은 영역은 무시
        if region.area < min_area:
            continue
        
        # 바운딩 박스 계산
        minr, minc, maxr, maxc = region.bbox
        
        # 결함 정보 저장
        defect = {
            'bbox': (minc, minr, maxc, maxr),
            'area': region.area,
            'centroid': region.centroid,
            'score': np.mean(segmentation_mask[region.coords[:, 0], region.coords[:, 1]])
        }
        defects.append(defect)
    
    return defects

def visualize_defects(image, defects):
    vis_image = image.copy()
    
    for defect in defects:
        x1, y1, x2, y2 = defect['bbox']
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, f"Score: {defect['score']:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_image

# 메인 실행 코드
if __name__ == "__main__":
    # 세그멘테이션 모델로부터 얻은 결과라고 가정
    segmentation_mask = cv2.imread('segmentation_result.png', cv2.IMREAD_GRAYSCALE) / 255.0
    original_image = cv2.imread('original_image.jpg')
    
    # 후처리 수행
    defects = post_process_segmentation(segmentation_mask)
    
    # 결과 시각화
    result_image = visualize_defects(original_image, defects)
    
    # 결과 저장
    cv2.imwrite('defect_detection_result.jpg', result_image)
    
    # 검출된 결함 정보 출력
    for i, defect in enumerate(defects):
        print(f"Defect {i+1}:")
        print(f"  Bounding Box: {defect['bbox']}")
        print(f"  Area: {defect['area']}")
        print(f"  Centroid: {defect['centroid']}")
        print(f"  Score: {defect['score']:.2f}")
        print()