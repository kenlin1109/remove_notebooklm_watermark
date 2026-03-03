import argparse
import cv2
import numpy as np
import random
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
MUSIC_EXTENSIONS = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg"}
DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = Path(__file__).parent / "assets" / "images"
MUSIC_DIR = Path(__file__).parent / "assets" / "music"
BGM_VOLUME = 0.01  # BGM 相對於原聲的音量比例（1.0 = 100%）

PROFILES: dict[str, dict[str, str]] = {
    "success": {
        "cover": "cover_success.jpg",
        "watermark": "watermark_success.png",
    },
    "ai": {
        "cover": "cover_ai.png",
        "watermark": "watermark_ai.png",
    },
}


def has_audio(video_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def pick_bgm() -> Path | None:
    if not MUSIC_DIR.exists():
        return None
    tracks = [p for p in MUSIC_DIR.iterdir() if p.suffix.lower() in MUSIC_EXTENSIONS]
    return random.choice(tracks) if tracks else None


def find_input_video() -> Path:
    videos = [
        p
        for p in DATA_DIR.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and not p.stem.endswith("_processed")
    ]
    if len(videos) == 0:
        raise FileNotFoundError("data/ 資料夾中找不到影片檔案")
    if len(videos) > 1:
        names = ", ".join(v.name for v in videos)
        raise ValueError(f"data/ 資料夾中有超過 1 個影片（{names}），請只保留 1 個")
    return videos[0]


def process_frame(frame: np.ndarray, watermark: np.ndarray) -> np.ndarray:
    result = frame.copy()
    result[660:680, 1105:1235] = watermark
    return result


def crop_frame(frame: np.ndarray) -> np.ndarray:
    """上下固定裁 5px，左右固定裁 10px。"""
    h, w = frame.shape[:2]
    return frame[5 : h - 5, 10 : w - 10]


def process_video(input_path: Path, profile: str) -> Path:
    output_path = input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"
    temp_path = input_path.parent / f"{input_path.stem}_temp{input_path.suffix}"

    assets = PROFILES[profile]
    watermark = cv2.imread(str(IMAGES_DIR / assets["watermark"]))
    watermark = cv2.resize(watermark, (1235 - 1105, 680 - 660))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cutoff_frame = total_frames - int(fps * 2.5)

    # 9:16 畫布
    canvas_w = src_w
    canvas_h = round(src_w * 16 / 9)
    if canvas_h % 2 != 0:
        canvas_h += 1

    # 載入並縮放 cover，與白底混合
    cover_src = cv2.imread(str(IMAGES_DIR / assets["cover"]))
    if cover_src is None:
        raise RuntimeError(f"找不到 cover 圖檔：{assets['cover']}")
    cover_h_scaled = round(canvas_w * cover_src.shape[0] / cover_src.shape[1])
    cover_resized = cv2.resize(cover_src, (canvas_w, cover_h_scaled))
    white = np.full_like(cover_resized, 255)
    cover = cv2.addWeighted(cover_resized, 0.6, white, 0.4, 0)

    video_area_h = canvas_h - cover_h_scaled * 2

    print(f"輸入影片：{input_path.name}  ({src_w}x{src_h})")
    print(
        f"輸出畫布：{canvas_w}x{canvas_h}（9:16），cover 高度：{cover_h_scaled}px，影片區域：{video_area_h}px"
    )
    print(f"輸出影片：{output_path.name}")
    print("處理中...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (canvas_w, canvas_h))

    frame_idx = 0
    try:
        while frame_idx < cutoff_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # 水印覆蓋
            frame = process_frame(frame, watermark)

            # 裁切黑邊
            cropped = crop_frame(frame)
            ch, cw = cropped.shape[:2]

            # 等比例縮放到畫布寬度
            scale = canvas_w / cw
            new_w = canvas_w
            new_h = min(int(ch * scale), video_area_h)
            resized = cv2.resize(cropped, (new_w, new_h))

            # 組合畫布
            canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
            canvas[0:cover_h_scaled, 0:canvas_w] = cover
            y_offset = cover_h_scaled + (video_area_h - new_h) // 2
            canvas[y_offset : y_offset + new_h, 0:new_w] = resized
            canvas[canvas_h - cover_h_scaled : canvas_h, 0:canvas_w] = cover
            out.write(canvas)

            frame_idx += 1
            if total_frames > 0 and frame_idx % max(1, cutoff_frame // 20) == 0:
                pct = frame_idx / cutoff_frame * 100
                print(f"  進度：{pct:.0f}% ({frame_idx}/{cutoff_frame})")
    finally:
        cap.release()
        out.release()

    bgm = pick_bgm()
    print("合併音訊中...")
    if bgm:
        print(f"背景音樂：{bgm.name}")
        if has_audio(input_path):
            # 原聲 + BGM 各自響度正規化後混音
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(temp_path),
                    "-i", str(input_path),
                    "-stream_loop", "-1", "-i", str(bgm),
                    "-filter_complex", f"[2:a]volume={BGM_VOLUME}[a2];[1:a][a2]amix=inputs=2:duration=shortest:normalize=0[aout]",
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(output_path),
                ],
                check=True,
            )
        else:
            # 無原聲，只用 BGM
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(temp_path),
                    "-stream_loop", "-1", "-i", str(bgm),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(output_path),
                ],
                check=True,
            )
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(temp_path),
                "-i", str(input_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
        )
    temp_path.unlink()

    print(f"完成！已儲存至：{output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="NotebookLM 浮水印移除工具")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        required=True,
        help="選擇素材組合：" + ", ".join(PROFILES.keys()),
    )
    args = parser.parse_args()

    try:
        input_path = find_input_video()
        process_video(input_path, args.profile)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
