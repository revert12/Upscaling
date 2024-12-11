#!/usr/bin/python3

import os
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import ffmpeg

# 동영상의 fps를 자동으로 추출하는 함수
def get_video_fps(video_path):
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='stream=r_frame_rate')
        fps_str = probe['streams'][0]['r_frame_rate']
        numerator, denominator = map(int, fps_str.split('/'))
        fps = numerator / denominator
        return fps
    except Exception as e:
        print(f"오류 발생: {e}")
        raise

# 동영상을 프레임별로 추출하는 함수
def extract_frames(video_path, output_folder, fps):
    try:
        os.makedirs(output_folder, exist_ok=True)
        # FFmpeg로 동영상에서 프레임 추출
        print(f"프레임 추출 중: {video_path} -> {output_folder}")
        ffmpeg.input(video_path).output(f'{output_folder}/frame_%05d.png', vf=f'fps={fps}').run()
    except Exception as e:
        print(f"프레임 추출 오류 발생: {e}")
        raise

# 프레임을 업스케일하는 함수
def upscale_frames(input_folder, output_folder, model_name="RealESRGAN_x2plus"):
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_block=23, scale=2)  # scale 2x로 변경
        model_path = f'Real-ESRGAN-master/weights/{model_name}.pth'
        if not os.path.exists(model_path):
            print(f"경고: 모델 파일이 존재하지 않습니다: {model_path}")
            return
        
        # 타일 크기를 줄여 메모리 사용량을 최적화
        upsampler = RealESRGANer(scale=2, model_path=model_path, model=model, tile=4, tile_pad=10, pre_pad=0)
        
        os.makedirs(output_folder, exist_ok=True)
        
        for frame_name in sorted(os.listdir(input_folder)):
            if frame_name.endswith(".png"):
                frame_path = os.path.join(input_folder, frame_name)
                img = cv2.imread(frame_path)
                if img is None:
                    print(f"경고: 이미지 로드 실패 - {frame_path}")
                    continue
                output, _ = upsampler.enhance(img, outscale=2)  # 2배 업스케일
                upscaled_frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(upscaled_frame_path, output)
    except Exception as e:
        print(f"프레임 업스케일 오류 발생: {e}")
        raise

# 업스케일된 프레임들을 다시 동영상으로 합치는 함수
def frames_to_video(input_folder, output_video_path, fps):
    try:
        frames = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.png')]
        if not frames:
            print(f"경고: 프레임 파일이 없습니다: {input_folder}")
            return
        
        first_frame = cv2.imread(os.path.join(input_folder, frames[0]))
        height, width, _ = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_path = os.path.join(input_folder, frame)
            frame_img = cv2.imread(frame_path)
            out.write(frame_img)
        
        out.release()
        print(f"동영상 생성 완료: {output_video_path}")
    except Exception as e:
        print(f"동영상 합치기 오류 발생: {e}")
        raise

# 메인 함수
def upscale_video(input_video_path, output_video_path, resolution="FHD (1920 x 1080)", temp_frame_folder="/vfs/upscaler/original_frames", upscaled_frame_folder="/vfs/upscaler/upscaled_frames"):
    try:
        # 동영상의 fps 추출
        fps = get_video_fps(input_video_path)
        print(f"입력 동영상의 fps: {fps}")
        
        # 동영상 크기 추출
        video_capture = cv2.VideoCapture(input_video_path)
        if not video_capture.isOpened():
            raise ValueError(f"동영상 열기 실패: {input_video_path}")
        
        video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        final_width = None
        final_height = None
        aspect_ratio = float(video_width / video_height)

        # 해상도에 따른 출력 크기 설정
        if resolution == "FHD (1920 x 1080)":
            final_width, final_height = 1920, 1080
        elif resolution == "2k (2560 x 1440)":
            final_width, final_height = 2560, 1440
        elif resolution == "4k (3840 x 2160)":
            final_width, final_height = 3840, 2160
        elif resolution == "720p":
            final_width, final_height = 1280, 720  # 720p 해상도로 다운스케일링
        elif resolution == "2 x original":
            final_width, final_height = 2 * video_width, 2 * video_height
        elif resolution == "3 x original":
            final_width, final_height = 3 * video_width, 3 * video_height
        elif resolution == "4 x original":
            final_width, final_height = 4 * video_width, 4 * video_height

        # 화면 비율에 맞춰 해상도 조정
        if aspect_ratio == 1.0 and "original" not in resolution:
            final_height = final_width
        if aspect_ratio < 1.0 and "original" not in resolution:
            temp = final_width
            final_width = final_height
            final_height = temp

        print(f"최종 해상도: {final_width}x{final_height}, 비율: {aspect_ratio}")

        # 동영상 프레임 추출
        print("동영상 프레임 추출 중...")
        extract_frames(input_video_path, temp_frame_folder, fps)
        
        # 프레임 업스케일링
        print("프레임 업스케일링 중...")
        upscale_frames(temp_frame_folder, upscaled_frame_folder)
        
        # 업스케일된 프레임으로 동영상 생성
        print("업스케일된 프레임으로 동영상 생성 중...")
        frames_to_video(upscaled_frame_folder, output_video_path, fps)
        
        print(f"업스케일링 완료! 결과는 {output_video_path}에 저장되었습니다.")
    
    except Exception as e:
        print(f"전체 프로세스에서 오류 발생: {e}")

# 스크립트를 직접 실행할 때 메인 함수 호출
if __name__ == "__main__":
    # 경로 수정 부분 (기존 경로에 맞춰 설정)
    input_video = '../Downloads/1280x720_15fps.mp4'  # 입력 비디오 경로 (이 부분만 수정)
    output_video = '../Downloads/output_1280x720_15fps.mp4'  # 출력 비디오 경로 (이 부분만 수정)
    
    upscale_video(input_video, output_video)
