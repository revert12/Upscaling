#!/bin/bash

# 1. GPU가 사용할 수 있는지 확인
python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not detected.. Please change runtime to GPU'"

# 2. 필요한 라이브러리 설치 (CUDA 11.8 버전에 맞춰서 PyTorch, torchvision 설치)
echo "Installing required libraries..."
pip install -q torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -q basicsr facexlib gfpgan ffmpeg ffmpeg-python
pip install -q -r requirements.txt

# 3. Real-ESRGAN 리포지토리 클론
echo "Cloning Real-ESRGAN repository..."
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN

# 4. 패키지 설정 (Python 패키지 설치 및 개발 모드로 설정)
echo "Setting up Real-ESRGAN..."
python3 setup.py develop

echo "Setup complete. Real-ESRGAN is now ready to use."
