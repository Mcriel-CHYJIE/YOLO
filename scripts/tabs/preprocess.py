"""视频预处理标签页 — 重命名+缩放+抽帧"""
from scripts.tabs.base import *
import cv2


class VideoPreprocessWorker(QThread):
    """视频预处理工作线程"""
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # current_video_idx, total_videos
    video_progress = pyqtSignal(int, int)  # current_frame, total_frames
    done = pyqtSignal(bool, str)
    image_saved = pyqtSignal(str)  # 发送保存的图片路径

    def __init__(self, src_folder: str, out_folder: str, target_size: int = 640, target_fps: float = 2.0):
        super().__init__()
        self.src_folder = Path(src_folder)
        self.out_folder = Path(out_folder)
        self.target_size = target_size
        self.target_fps = target_fps
        self._stop = False

    def stop(self):
        self._stop = True

    @staticmethod
    def _letterbox_resize(img, size=640):
        """保持宽高比缩放到目标尺寸，不足补黑边"""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        # 使用INTER_AREA进行缩小（减少混叠），INTER_CUBIC进行放大（更高质量）
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
        out = np.zeros((size, size, 3), dtype=np.uint8)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = resized
        return out

    def run(self):
        try:
            import random
            # ── 1. 获取视频文件 ──
            exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
            videos = sorted([f for f in self.src_folder.iterdir()
                             if f.suffix.lower() in exts and f.is_file()])
            if not videos:
                self.log.emit(' 未找到视频文件')
                self.done.emit(False, 'No video files found')
                return

            n = len(videos)
            folder_name = self.src_folder.name  # 获取文件夹名称
            self.log.emit(f' {self.src_folder.name} — {n} 个视频')
            self.out_folder.mkdir(parents=True, exist_ok=True)

            # ── 2. 第一阶段：统一重命名所有视频 ──
            self.log.emit(' 开始重命名视频...')
            renamed_videos = []
            for idx, src_path in enumerate(videos):
                if self._stop:
                    self.log.emit(' 已停止')
                    self.done.emit(False, 'Stopped')
                    return

                # 重命名：00, 01, 02...
                new_name = f'{idx:02d}{src_path.suffix}'
                renamed_path = self.src_folder / new_name

                # 若文件名不同则重命名
                if src_path.name != new_name:
                    if renamed_path.exists():
                        self.log.emit(f' 已存在，跳过重命名')
                        renamed_videos.append(renamed_path)
                    else:
                        src_path.rename(renamed_path)
                        self.log.emit(f' {src_path.name} → {new_name}')
                        renamed_videos.append(renamed_path)
                else:
                    self.log.emit(f' {new_name}')
                    renamed_videos.append(renamed_path)

            self.log.emit(f' 重命名完成，共 {len(renamed_videos)} 个视频')

            # ── 3. 第二阶段：统一抽帧（每秒1帧，随机帧） ──
            self.log.emit(' 开始抽帧处理（每秒1帧，随机帧）...')
            for idx, renamed_path in enumerate(renamed_videos):
                if self._stop:
                    self.log.emit(' 已停止')
                    self.done.emit(False, 'Stopped')
                    return

                video_num = f'{idx:02d}'  # 视频序号
                new_name = renamed_path.name

                # 打开视频抽帧
                cap = cv2.VideoCapture(str(renamed_path))
                if not cap.isOpened():
                    self.log.emit(f' 无法打开 {new_name}，跳过')
                    continue

                src_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_seconds = int(total_frames / src_fps)  # 视频总秒数
                frames_per_second = int(src_fps)  # 每秒帧数

                self.log.emit(f' {new_name} | {src_fps:.1f}fps | '
                              f'{total_frames}帧 | 约{total_seconds}秒 → 约{total_seconds}帧')

                self.progress.emit(idx + 1, n)

                saved = 0
                # 遍历每一秒
                for second in range(total_seconds):
                    if self._stop:
                        break
                    
                    # 计算该秒的起始和结束帧
                    start_frame = second * frames_per_second
                    end_frame = min((second + 1) * frames_per_second, total_frames)
                    
                    # 随机选择该秒中的一帧
                    random_frame = random.randint(start_frame, end_frame - 1)
                    
                    # 跳转到随机帧
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                    ret, frame = cap.read()
                    
                    if ret:
                        # 缩放到640×640
                        resized = self._letterbox_resize(frame, self.target_size)
                        # 命名格式：文件夹名-视频序号-秒数.jpg
                        out_filename = f'{folder_name}-{video_num}-{second:04d}.jpg'
                        out_path = self.out_folder / out_filename
                        cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 98, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
                        saved += 1
                        self.video_progress.emit(saved, total_seconds)
                        # 发送图片路径用于预览
                        self.image_saved.emit(str(out_path))

                cap.release()
                self.log.emit(f' {new_name} → {saved} 帧（{total_seconds}秒）')

            self.log.emit(f' 全部完成！共处理 {n} 个视频')
            self.done.emit(True, f'Complete — {n} videos processed')

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log.emit(f' {e}')
            self.done.emit(False, str(e))


class PreprocessTab(QWidget):
    """视频预处理标签页"""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._worker = None
        self._build()
        self._refresh_folder_list()

    @property
    def _before_root(self):
        return ROOT / 'original' / 'before'

    @property
    def _after_root(self):
        return ROOT / 'original' / 'after'

    # ═══════════════ BUILD UI ═══════════════

    def _build(self):
        lo = QHBoxLayout(self)
        lo.setContentsMargins(8, 8, 8, 8)
        lo.setSpacing(8)

        # ─ 左栏 ──
        left = QWidget()
        left.setFixedWidth(300)
        ll = QVBoxLayout(left)
        ll.setSpacing(6)
        ll.setContentsMargins(0, 0, 0, 0)

        # -- Group 1: 输入输出目录 --
        g1 = QGroupBox('Directories')
        g1l = QVBoxLayout(g1)
        g1l.setSpacing(8)
        g1l.setContentsMargins(10, 14, 10, 10)

        # Input 输入目录
        il = QHBoxLayout()
        il.setSpacing(6)
        il.addWidget(QLabel('Input', styleSheet=f'font-size:10px;color:{TEXT2};font-weight:500;min-width:40px;'))
        self.src_input = QLineEdit()
        self.src_input.setMinimumHeight(24)
        self.src_input.setReadOnly(True)
        il.addWidget(self.src_input, 1)
        src_browse_btn = QPushButton('...')
        src_browse_btn.setFixedSize(30, 26)
        src_browse_btn.setToolTip('Browse input directory')
        src_browse_btn.clicked.connect(self._browse_input)
        il.addWidget(src_browse_btn)
        g1l.addLayout(il)

        self.input_label = QLabel()
        self.input_label.setStyleSheet(f'font-size:10px;color:{TEXT3};padding:2px 4px;background:{BG};border-radius:3px;')
        self.input_label.setWordWrap(True)
        g1l.addWidget(self.input_label)

        # Output 输出目录
        ol = QHBoxLayout()
        ol.setSpacing(6)
        ol.addWidget(QLabel('Output', styleSheet=f'font-size:10px;color:{TEXT2};font-weight:500;min-width:40px;'))
        self.out_input = QLineEdit()
        self.out_input.setMinimumHeight(24)
        self.out_input.setReadOnly(True)
        ol.addWidget(self.out_input, 1)
        out_browse_btn = QPushButton('...')
        out_browse_btn.setFixedSize(30, 26)
        out_browse_btn.setToolTip('Browse output directory')
        out_browse_btn.clicked.connect(self._browse_output)
        ol.addWidget(out_browse_btn)
        g1l.addLayout(ol)
        
        # Reset 重置按钮
        reset_btn = QPushButton('⟲ Reset to Default')
        reset_btn.setMinimumHeight(24)
        reset_btn.setToolTip('Reset to default directories')
        reset_btn.clicked.connect(self._refresh_folder_list)
        g1l.addWidget(reset_btn)
        ll.addWidget(g1)

        # -- Group 2: 视频列表 --
        g2 = QGroupBox('Videos')
        g2l = QVBoxLayout(g2)
        g2l.setSpacing(6)
        g2l.setContentsMargins(10, 14, 10, 10)

        self.video_list = QListWidget()
        self.video_list.setStyleSheet(f'''
            QListWidget {{
                background: {BG};
                border: 1px solid {BORDER};
                border-radius: 4px;
                font-size: 11px;
                color: {TEXT};
                padding: 3px;
            }}
            QListWidget::item {{
                padding: 4px 6px;
                border-radius: 3px;
            }}
            QListWidget::item:selected {{
                background: {PRI};
                color: #fff;
            }}
        ''')
        self.video_list.setMinimumHeight(140)
        g2l.addWidget(self.video_list, 1)

        self.video_count = QLabel('0 videos')
        self.video_count.setStyleSheet(f'font-size:10px;color:{TEXT3};padding:2px 0;')
        g2l.addWidget(self.video_count)
        ll.addWidget(g2)

        # -- Group 3: 参数 --
        g3 = QGroupBox('Parameters')
        g3l = QGridLayout(g3)
        g3l.setSpacing(6)
        g3l.setContentsMargins(10, 14, 10, 10)
        g3l.setColumnStretch(0, 1)
        g3l.setColumnStretch(1, 1)

        self.out_fps = QDoubleSpinBox()
        self.out_fps.setRange(0.5, 30)
        self.out_fps.setValue(2.0)
        self.out_fps.setSingleStep(1)
        self.out_fps.setDecimals(1)
        self.out_fps.setMinimumHeight(24)
        self._add_param(g3l, 'FPS', self.out_fps, 0, 0)

        self.target_size = QSpinBox()
        self.target_size.setRange(256, 1920)
        self.target_size.setValue(640)
        self.target_size.setSingleStep(64)
        self.target_size.setMinimumHeight(24)
        self._add_param(g3l, 'Size', self.target_size, 0, 1)

        ll.addWidget(g3)

        # -- Group 4: Control --
        g4 = QGroupBox('Control')
        g4l = QVBoxLayout(g4)
        g4l.setSpacing(6)
        g4l.setContentsMargins(10, 14, 10, 10)

        br = QHBoxLayout()
        br.setSpacing(6)
        self.start_btn = QPushButton('▶ Start')
        self.start_btn.setObjectName('pri')
        self.start_btn.setMinimumHeight(28)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton('⏹ Stop')
        self.stop_btn.setObjectName('danger')
        self.stop_btn.setMinimumHeight(28)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        br.addWidget(self.start_btn, 1)
        br.addWidget(self.stop_btn, 1)
        g4l.addLayout(br)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(6)
        self.progress_bar.setTextVisible(False)
        g4l.addWidget(self.progress_bar)

        # 进度统计卡片
        sr = QHBoxLayout()
        sr.setSpacing(6)
        for lbl, attr, col in [('Video', '_pv', TEXT), ('Frame', '_pf', GREEN)]:
            cw = QWidget()
            cw.setStyleSheet(f'background:{BG};border-radius:5px;')
            cl2 = QVBoxLayout(cw)
            cl2.setContentsMargins(8, 6, 8, 6)
            cl2.setSpacing(2)
            v = QLabel('—')
            v.setStyleSheet(f'font-size:16px;font-weight:700;color:{col};qproperty-alignment:AlignCenter;')
            setattr(self, attr, v)
            cl2.addWidget(v)
            cl2.addWidget(QLabel(lbl, styleSheet=f'font-size:9px;color:{TEXT3};font-weight:500;qproperty-alignment:AlignCenter;'))
            sr.addWidget(cw, 1)
        g4l.addLayout(sr)

        self.status_label = QLabel('Ready')
        self.status_label.setStyleSheet(f'font-size:10px;color:{TEXT2};padding:3px 0;')
        g4l.addWidget(self.status_label)
        ll.addWidget(g4)

        ll.addStretch()
        lo.addWidget(left)

        # ── 右栏：图片预览 + 日志 ──
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        # 图片预览区
        preview_group = QGroupBox('Image Preview')
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 12, 8, 8)
        preview_layout.setSpacing(6)
        
        # 预览图片显示区域
        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(150, 150)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(f'''QLabel {{
            background: {BG};
            border: 1px solid {BORDER};
            border-radius: 4px;
            color: {TEXT3};
            font-size: 12px;
        }}''')
        self.preview_label.setText('—')
        preview_layout.addWidget(self.preview_label, 1)
        
        # 统计信息
        self.preview_stats = QLabel('0 images processed')
        self.preview_stats.setStyleSheet(f'font-size:9px;color:{TEXT3};padding:2px 0;')
        preview_layout.addWidget(self.preview_stats)
        
        rl.addWidget(preview_group, 1)  # 添加拉伸因子，填充空间

        # 日志区
        log_header = QWidget()
        log_header.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:6px 6px 0 0;')
        hl = QHBoxLayout(log_header)
        hl.setContentsMargins(12, 6, 12, 6)
        hl.addWidget(QLabel('● Console', styleSheet=f'font-size:11px;font-weight:600;color:{TEXT3};'))
        hl.addStretch()
        clear_btn = QPushButton('Clear')
        clear_btn.clicked.connect(self._clear_log)
        clear_btn.setStyleSheet('padding:3px 10px;min-height:20px;font-size:10px;border-radius:3px;')
        hl.addWidget(clear_btn)
        rl.addWidget(log_header)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(f'''
            QTextEdit {{
                background: {CON};
                color: {CON_T};
                border: 1px solid {BORDER};
                border-top: none;
                border-radius: 0 0 6px 6px;
                padding: 8px 12px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
            }}
        ''')
        self.log_output.setFixedHeight(100)
        rl.addWidget(self.log_output)
        lo.addWidget(right, 1)

    def _add_param(self, grid, label, widget, r, c):
        cw = QWidget()
        cw.setStyleSheet(f'background:{BG};border-radius:4px;')
        cl = QHBoxLayout(cw)
        cl.setContentsMargins(8, 2, 8, 2)
        cl.setSpacing(6)
        cl.addWidget(QLabel(label, styleSheet=f'font-size:10px;color:{TEXT};font-weight:500;'))
        cl.addWidget(widget, 1)
        grid.addWidget(cw, r, c)

    # ═══════════════ LOGIC ═══════════════

    def _browse_input(self):
        """选择输入目录"""
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Input Directory',
            str(self._before_root) if self._before_root.exists() else str(ROOT)
        )
        if folder:
            folder_path = Path(folder)
            self.src_input.setText(str(folder_path))
            self.input_label.setText(f'📂 {folder_path}')
            # 自动设置对应的输出目录
            folder_name = folder_path.name
            default_output = self._after_root / folder_name
            self.out_input.setText(str(default_output))
            self._refresh_video_list(folder_path)

    def _browse_output(self):
        """选择输出目录"""
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Output Directory',
            str(self._after_root) if self._after_root.exists() else str(ROOT)
        )
        if folder:
            folder_path = Path(folder)
            self.out_input.setText(str(folder_path))

    def _refresh_folder_list(self):
        """重置为默认目录"""
        if self._before_root.exists():
            # 获取所有子文件夹
            dirs = sorted([d.name for d in self._before_root.iterdir() if d.is_dir()])
            if dirs:
                # 默认选择第一个
                default_folder = dirs[0]
                src = self._before_root / default_folder
                out = self._after_root / default_folder
                self.src_input.setText(str(src))
                self.out_input.setText(str(out))
                self.input_label.setText(f' {src}')
                self._refresh_video_list(src)
                return
        
        self.src_input.clear()
        self.out_input.clear()
        if not self._before_root.exists():
            self.input_label.setText(f'❌ 目录不存在: {self._before_root}')
            self.video_list.clear()
            self.video_count.setText('0 videos')
            return

        dirs = sorted([d.name for d in self._before_root.iterdir() if d.is_dir()])
        if not dirs:
            self.input_label.setText('⚠️ 没有子文件夹')

    def _on_folder_selected(self):
        """选中文件夹时刷新视频列表（保留用于兼容）"""
        folder = self.src_combo.currentText()
        if not folder:
            return
        src = self._before_root / folder
        out = self._after_root / folder
        self.src_input.setText(str(src))
        self.out_input.setText(str(out))
        self.input_label.setText(f'📂 {src}')
        self._refresh_video_list(src)

    def _refresh_video_list(self, folder: Path):
        """列出文件夹中的视频文件"""
        self.video_list.clear()
        exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        videos = sorted([f.name for f in folder.iterdir()
                         if f.suffix.lower() in exts and f.is_file()])
        if videos:
            for v in videos:
                self.video_list.addItem(v)
        self.video_count.setText(f'{len(videos)} videos')

    def _start(self):
        src_folder_str = self.src_input.text()
        out_folder_str = self.out_input.text()
        
        if not src_folder_str:
            QMessageBox.warning(self, 'Warning', 'Please select an input directory first')
            return
        if not out_folder_str:
            QMessageBox.warning(self, 'Warning', 'Please select an output directory first')
            return
        
        src_folder = Path(src_folder_str)
        out_folder = Path(out_folder_str)
        
        if not src_folder.exists():
            QMessageBox.warning(self, 'Warning', f'Input directory does not exist: {src_folder}')
            return

        exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        videos = [f for f in src_folder.iterdir() if f.suffix.lower() in exts and f.is_file()]
        if not videos:
            QMessageBox.warning(self, 'Warning', 'No video files found in this folder')
            return

        # 锁定UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.src_input.setEnabled(False)
        self.out_input.setEnabled(False)
        self.out_fps.setEnabled(False)
        self.target_size.setEnabled(False)
        self.progress_bar.setValue(0)
        self._pv.setText('—')
        self._pf.setText('—')
        self._clear_log()

        self._log(f'🚀 Starting preprocessing: {src_folder.name}')
        self._log(f' Input:  {src_folder}')
        self._log(f'📁 Output: {out_folder}')
        self._log(f'   Size: {self.target_size.value()}×{self.target_size.value()}')
        self._log(f'   FPS:  {self.out_fps.value()}')

        self._worker = VideoPreprocessWorker(
            src_folder=str(src_folder),
            out_folder=str(out_folder),
            target_size=self.target_size.value(),
            target_fps=self.out_fps.value(),
        )
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_video_progress)
        self._worker.video_progress.connect(self._on_frame_progress)
        self._worker.image_saved.connect(self._on_image_saved)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.stop_btn.setEnabled(False)
            self._log('⏳ Stopping...')

    def _on_video_progress(self, current, total):
        self.progress_bar.setValue(int(current / total * 100))
        self._pv.setText(f'{current}/{total}')

    def _on_frame_progress(self, current, total):
        self._pf.setText(f'{current}')

    def _on_image_saved(self, image_path: str):
        """处理保存的图片预览"""
        try:
            from PyQt5.QtGui import QPixmap
            # 更新统计信息
            current_count = int(self.preview_stats.text().split()[0]) + 1
            self.preview_stats.setText(f'{current_count} images processed')
            
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.preview_label.width(), 
                    self.preview_label.height(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.preview_label.setPixmap(scaled_pixmap)
                self.preview_label.setText('')
        except Exception as e:
            pass

    def _on_done(self, ok, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.src_input.setEnabled(True)
        self.out_input.setEnabled(True)
        self.out_fps.setEnabled(True)
        self.target_size.setEnabled(True)
        self.progress_bar.setValue(100 if ok else 0)
        self._log(f'{"🎉" if ok else "❌"} {msg}')
        self.status_label.setText(msg)
        self._worker = None

    def _log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        color = CON_T
        if '✅' in msg or '🎉' in msg: color = GREEN
        elif '❌' in msg: color = RED
        elif '⚠️' in msg: color = AMBER
        elif '🚀' in msg: color = '#a5b4fc'
        elif '📄' in msg or '📁' in msg: color = TEXT3
        elif '🎬' in msg: color = PRI
        html = f'<span style="color:#6b7280">[{ts}]</span> <span style="color:{color}">{msg}</span>'
        self.log_output.append(html)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum())

    def _clear_log(self):
        self.log_output.clear()
