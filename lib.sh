#!/bin/bash

# 1. Python 및 pip 버전 확인
echo "Python version: $(python3 --version)"
echo "pip version: $(pip3 --version)"

# 2. CUDA와 호환되는 PyTorch와 torchvision 설치
echo "Installing PyTorch and torchvision..."
pip3 install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Real-ESRGAN 리포지토리 클론
echo "Cloning Real-ESRGAN repository..."
# git clone https://github.com/xinntao/Real-ESRGAN.git

# 다운로드할 레포지토리 URL
repo_url="https://github.com/xinntao/Real-ESRGAN/archive/refs/heads/master.zip"

# 저장할 파일 이름
zip_file="Real-ESRGAN-master.zip"

# ZIP 파일 다운로드
echo "Downloading $repo_url..."
wget -O $zip_file $repo_url

# 압축 해제
echo "Extracting $zip_file..."
unzip -o $zip_file

# 압축 해제 후 다운로드된 ZIP 파일 삭제 (원하는 경우)
rm $zip_file
cd Real-ESRGAN-master

# 4. 필요한 패키지 설치
echo "Installing required libraries..."
pip3 install -r requirements.txt

# 5. 개발 환경 설정
echo "Setting up Real-ESRGAN..."
 python3 setup.py develop  --user

echo "Setup complete. Real-ESRGAN is now ready to use."
