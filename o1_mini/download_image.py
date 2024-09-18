import os
import requests

def download_images(url_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, url in enumerate(url_list):
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(save_dir, f'image_{idx}.png')
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f'다운로드 완료: {file_path}')
        else:
            print(f'다운로드 실패: {url}')

if __name__ == "__main__":
    urls = [
        'https://example.com/image1.png',
        'https://example.com/image2.png',
        # 추가 이미지 URL
    ]
    download_images(urls, 'data/raw_images')