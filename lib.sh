# Python 라이브러리 설치
pip install torch torchvision opencv-python-headless ffmpeg-python tqdm

# Real-ESRGAN 레포지토리 클론 및 요구 사항 설치
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop

# FFmpeg 설치
# 윈도우에서는 FFmpeg를 https://ffmpeg.org/download.html에서 다운로드하고 PATH에 추가
# macOS에서는: brew install ffmpeg
# 리눅스에서는: sudo apt-get install ffmpeg
