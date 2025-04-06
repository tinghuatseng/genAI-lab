# IG/ Youtube 影片轉文字

## 啟動 MCP Server

```shell
uvicorn mcp_server:app
```

## 下載 Youtube/ IG 影片
```
 curl -X 'POST' 'http://127.0.0.1:8000/transcribe_ig' -H 'Content-Type: application/json' -d '{"video_url": "https://www.instagram.com/reel/DEhWqiAT__2/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA=="}'
 ```
 