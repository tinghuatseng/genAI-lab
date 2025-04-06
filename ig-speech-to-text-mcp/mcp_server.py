from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import whisper
import yt_dlp
import os
import cv2
import easyocr

# 設定 FastAPI 伺服器
app = FastAPI()

# 設定下載路徑
DOWNLOAD_FOLDER = "downloads"
AUDIO_FILE = "audio.wav"

# 初始化 Whisper 語音轉文字模型
model = whisper.load_model("base")

# 確保下載資料夾存在
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def perform_ocr(image_path, languages):
    """
    Performs OCR on an image.

    Args:
        image_path (str): Path to the image file.
        languages (list): List of languages to use for OCR.

    Returns:
        str: The extracted text.
    """
    reader = easyocr.Reader(languages,gpu="cuda:0")
    results = reader.readtext(image_path)
    text = " ".join([result[1] for result in results])
    return text

def extract_frames(video_path, interval):
    """
    Extracts frames from a video at a specified interval.

    Args:
        video_path (str): Path to the video file.
        interval (int): Interval between frames in seconds.

    Returns:
        list: A list of paths to the extracted frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0, frame_count, int(interval * fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_path = f"downloads/frame_{i}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        else:
            break
    cap.release()
    return frames

class VideoRequest(BaseModel):
    video_url: str

def download_video(video_url):
    """ 下載 IG 影片 """
    video_path = f"{DOWNLOAD_FOLDER}/video.mp4"

    ydl_opts = {
        'outtmpl': f"{DOWNLOAD_FOLDER}/video.mp4",
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
    }
    # 刪除舊檔案，確保新影片被下載
    if os.path.exists(video_path):
        os.remove(video_path)

    # with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #     ydl.download([video_url])
     # 使用 `yt-dlp` 擷取影片 & 貼文文字
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)  # `download=True` 表示實際下載

    # 獲取貼文內容
    post_caption = info.get("description", "").strip()  # 貼文標題/內文
    return video_path, post_caption

def extract_audio(video_path, output_audio_path):
    output_audio_path = f"{DOWNLOAD_FOLDER}/{output_audio_path}"
    """ 從影片擷取音訊 """
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)

    command = [
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_audio_path
    ]
    subprocess.run(command, check=True)
    return output_audio_path

def transcribe_audio(audio_path):
    """ 使用 Whisper 進行語音轉文字 """
    result = model.transcribe(audio_path)
    return result["text"]

@app.post("/transcribe_ig")
async def transcribe_ig(request: VideoRequest):
    """ API 端點，從 JSON 請求體讀取 IG 影片 URL，並進行轉換 """
    try:
        video_path, post_caption = download_video(request.video_url)
        audio_path = extract_audio(video_path, AUDIO_FILE)
        transcript = transcribe_audio(audio_path)
        frames = extract_frames(video_path, interval=5)
        ocr_text = ""
        for frame_path in frames:
            ocr_text += perform_ocr(frame_path, languages=['ch_tra', 'en']) + " "
        return {"status": "success", "transcript": transcript, "post_caption": post_caption, "ocr_text": ocr_text}  # 貼文內文}
    except Exception as e:
        return {"status": "error", "message": str(e)}