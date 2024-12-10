#!/bin/bash

# 1. Python 및 pip 버전 확인
echo "Python version: $(python3 --version)"
echo "pip version: $(pip3 --version)"

# 2. CUDA와 호환되는 PyTorch와 torchvision 설치
echo "Installing PyTorch and torchvision..."
pip3 install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# 3. Real-ESRGAN 리포지토리 클론
echo "Cloning Real-ESRGAN repository..."
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

# 4. 필요한 패키지 설치
echo "Installing required libraries..."
pip3 install -r requirements.txt

# 5. 개발 환경 설정
echo "Setting up Real-ESRGAN..."
 python3 setup.py develop  --user

echo "Setup complete. Real-ESRGAN is now ready to use."
