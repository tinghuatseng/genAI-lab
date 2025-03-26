from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import whisper
import yt_dlp
import os

# 設定 FastAPI 伺服器
app = FastAPI()

# 設定下載路徑
DOWNLOAD_FOLDER = "downloads"
AUDIO_FILE = "audio.wav"

# 初始化 Whisper 語音轉文字模型
model = whisper.load_model("base")

# 確保下載資料夾存在
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

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
        return {"status": "success", "transcript": transcript, "post_caption": post_caption}  # 貼文內文}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
