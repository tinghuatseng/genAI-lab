
import yt_dlp
import os

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

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return video_path