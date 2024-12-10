import os
import ffmpeg
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np

# 1. 동영상의 fps를 자동으로 추출하는 함수
def get_video_fps(video_path):
    probe = ffmpeg.probe(video_path, v="error", select_streams="v:0", show_entries="stream=r_frame_rate")
    fps_str = probe["streams"][0]["r_frame_rate"]
    numerator, denominator = map(int, fps_str.split("/"))
    fps = numerator / denominator
    return fps

# 2. 동영상을 프레임별로 추출하는 함수
def extract_frames(video_path, output_folder, fps):
    os.makedirs(output_folder, exist_ok=True)
    # FFmpeg로 동영상에서 프레임 추출
    ffmpeg.input(video_path).output(f'{output_folder}/frame_%04d.png', vf=f'fps={fps}').run()

# 3. 프레임을 업스케일하는 함수
def upscale_frames(input_folder, output_folder, model_name="RealESRGAN_x4plus"):
    # Real-ESRGAN 모델 설정
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_features=64, num_block=23, scale=4)
    
    # 모델 경로는 사용자 설정이 필요함 (예: 모델 가중치 파일 경로)
    model_path = f'weights/{model_name}.pth'
    
    # RealESRGANer 인스턴스를 생성
    upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 이미지 폴더 내 모든 이미지 처리
    for frame_name in sorted(os.listdir(input_folder)):
        if frame_name.endswith(".png"):
            frame_path = os.path.join(input_folder, frame_name)
            img = cv2.imread(frame_path)
            
            # 이미지를 모델로 업스케일
            output, _ = upsampler.enhance(img, outscale=4)
            
            # 업스케일된 이미지 저장
            upscaled_frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(upscaled_frame_path, output)

# 4. 업스케일된 프레임들을 다시 동영상으로 합치는 함수
def frames_to_video(input_folder, output_video_path, fps):
    # 프레임들을 정렬하여 동영상 생성
    frames = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.png')]
    
    # 첫 번째 프레임을 읽어서 크기 설정
    first_frame = cv2.imread(os.path.join(input_folder, frames[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 프레임들을 동영상에 추가
    for frame in frames:
        frame_path = os.path.join(input_folder, frame)
        frame_img = cv2.imread(frame_path)
        out.write(frame_img)
    
    out.release()

# 5. 메인 함수
def upscale_video(input_video_path, output_video_path, temp_frame_folder="frames", upscaled_frame_folder="upscaled_frames"):
    # 동영상의 fps 추출
    fps = get_video_fps(input_video_path)
    print(f"입력 동영상의 fps: {fps}")
    
    print("동영상 프레임 추출 중...")
    extract_frames(input_video_path, temp_frame_folder, fps)
    
    print("프레임 업스케일링 중...")
    upscale_frames(temp_frame_folder, upscaled_frame_folder)
    
    print("업스케일된 프레임으로 동영상 생성 중...")
    frames_to_video(upscaled_frame_folder, output_video_path, fps)
    
    print(f"업스케일링 완료! 결과는 {output_video_path}에 저장되었습니다.")

# 실행 예시
input_video = "../Downloads/seoul_park_30m.mp4"
output_video = "../Downloads/seoul_park_30m_up.mp4"
upscale_video(input_video, output_video)
