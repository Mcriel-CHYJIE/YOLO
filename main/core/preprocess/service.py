"""预处理业务逻辑 — 视频抽帧 Worker"""
import os, random, cv2
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from main.core.base import ROOT, VIDEO_EXTS

os.environ['OPENCV_FFMPEG_LOGLEVEL'] = 'error'


class VideoPreprocessWorker(QThread):
    """后台视频预处理线程：重命名+缩放+抽帧"""
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    video_progress = pyqtSignal(int, int)
    done = pyqtSignal(bool, str)
    image_saved = pyqtSignal(str)

    def __init__(self, src_folder: str, out_folder: str, target_size: int = 640):
        super().__init__()
        self.src_folder = Path(src_folder)
        self.out_folder = Path(out_folder)
        self.target_size = target_size
        self._stop = False

    def stop(self):
        self._stop = True

    @staticmethod
    def _resize_image(img, target_size=640):
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(img, (nw, nh), interpolation=interpolation)

    def run(self):
        try:
            exts = VIDEO_EXTS
            videos = sorted([f for f in self.src_folder.iterdir()
                             if f.suffix.lower() in exts and f.is_file()])
            if not videos:
                self.log.emit('未找到视频文件')
                self.done.emit(False, 'No video files found'); return
            n = len(videos)
            folder_name = self.src_folder.name
            self.log.emit(f'{self.src_folder.name} — {n} 个视频')
            self.out_folder.mkdir(parents=True, exist_ok=True)
            self.log.emit('开始重命名视频...')
            renamed_videos = []
            for idx, src_path in enumerate(videos):
                if self._stop: self.log.emit('已停止'); self.done.emit(False, 'Stopped'); return
                new_name = f'{idx:02d}{src_path.suffix}'
                rp = self.src_folder / new_name
                if src_path.name != new_name:
                    if rp.exists():
                        renamed_videos.append(rp)
                    else:
                        src_path.rename(rp); renamed_videos.append(rp)
                else:
                    renamed_videos.append(rp)
            self.log.emit(f'重命名完成，共 {len(renamed_videos)} 个视频')
            self.log.emit('开始抽帧处理（每秒1帧，随机帧）...')
            for idx, rp in enumerate(renamed_videos):
                if self._stop: self.log.emit('已停止'); self.done.emit(False, 'Stopped'); return
                vnum = f'{idx:02d}'; nm = rp.name
                cap = cv2.VideoCapture(str(rp))
                if not cap.isOpened(): self.log.emit(f'无法打开 {nm}，跳过'); continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                secs = int(total / fps)
                self.log.emit(f'{nm} | {fps:.1f}fps | {total}帧 | 约{secs}秒')
                self.progress.emit(idx + 1, n)
                saved = 0; errs = 0
                for s in range(secs):
                    try:
                        sf = s * int(fps); ef = min((s + 1) * int(fps), total)
                        rf = random.randint(sf, ef - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, rf)
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            resized = self._resize_image(frame, self.target_size)
                            fn = f'{folder_name}-{vnum}-{s:04d}.jpg'
                            cv2.imwrite(str(self.out_folder / fn), resized,
                                        [cv2.IMWRITE_JPEG_QUALITY, 98])
                            saved += 1
                            self.video_progress.emit(saved, secs)
                            self.image_saved.emit(str(self.out_folder / fn))
                        else:
                            errs += 1
                            if errs <= 3: self.log.emit(f'Frame decode failed at second {s}')
                    except Exception as e:
                        errs += 1
                        if errs <= 3: self.log.emit(f'Error at {s}: {str(e)[:50]}')
                cap.release()
                self.log.emit(f'{nm} → {saved} 帧{" (" + str(errs) + " decode errors)" if errs else ""}')
            self.log.emit(f'全部完成！共处理 {n} 个视频')
            self.done.emit(True, f'Complete — {n} videos processed')
        except Exception as e:
            import traceback; traceback.print_exc()
            self.log.emit(f'{e}'); self.done.emit(False, str(e))
