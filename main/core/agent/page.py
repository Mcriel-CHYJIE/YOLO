# =============================================================================
# YOLO Training Studio — AI Agent 标签页
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# =============================================================================

"""AI Agent 标签页 — YOLO 专家助手聊天"""

from pathlib import Path
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QTextDocument
from PyQt5.QtCore import QUrl, QSize, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from main.core.base import *
from . import service as svc


# ── 气泡常量 ──
_GREEN = '#07C160'
_GREEN_BORDER = '#06ad56'
_GRAY = '#f0f0f0'
_GRAY_BORDER = '#d8d8d8'
_BUBBLE_FONT = '14px'
_BUBBLE_PAD = '10px 16px'
_BUBBLE_LH = '1.6'
_MAX_W = 0  # 后续改为百分比


class _UserBubble(QWidget):
    """用户消息气泡（灰色，右对齐，无头像）"""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 3, 0, 3)

        lo = QHBoxLayout(self)
        lo.setContentsMargins(12, 0, 12, 0)
        lo.addStretch()

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(536)
        lbl.setStyleSheet(
            f"QLabel{{"
            f"background:{_GRAY};color:#1a1a1a;"
            f"border:1px solid {_GRAY_BORDER};"
            f"border-radius:16px 4px 16px 16px;"
            f"padding:{_BUBBLE_PAD};"
            f"font-size:{_BUBBLE_FONT};line-height:{_BUBBLE_LH};"
            f"}}")
        lo.addWidget(lbl)


class _AssistantBubble(QWidget):
    """助手消息气泡（绿色，左对齐，带 YOLO 头像）"""

    def __init__(self, text, avatar_pixmap, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 3, 0, 3)

        lo = QHBoxLayout(self)
        lo.setContentsMargins(12, 0, 12, 0)

        # 头像
        if not avatar_pixmap.isNull():
            ava_container = QWidget()
            ava_container.setStyleSheet('background:transparent;')
            ava_lo = QVBoxLayout(ava_container)
            ava_lo.setContentsMargins(0, 0, 0, 0)
            ava_lo.setSpacing(0)
            ava = QLabel()
            ava.setPixmap(avatar_pixmap)
            ava.setFixedSize(40, 40)
            ava.setStyleSheet('border-radius:6px;')
            ava_lo.addWidget(ava)
            ava_lo.addStretch()
            lo.addWidget(ava_container)
            lo.addSpacing(8)
        else:
            lo.addSpacing(48)

        # 气泡
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(536)
        lbl.setStyleSheet(
            f"QLabel{{"
            f"background:{_GREEN};color:#fff;"
            f"border:1px solid {_GREEN_BORDER};"
            f"border-radius:4px 16px 16px 16px;"
            f"padding:{_BUBBLE_PAD};"
            f"font-size:{_BUBBLE_FONT};line-height:{_BUBBLE_LH};"
            f"}}")
        lo.addWidget(lbl)
        lo.addStretch()


class AgentTab(QWidget):
    _reply_signal = pyqtSignal(str, str)  # (type, text) type='reply'|'error'
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self._messages = []
        self._thread = None
        self._loading = False
        self._build_ui()
        self._reset_chat()

    # ═══════════════════════════════════════════
    # UI Construction
    # ═══════════════════════════════════════════

    def _build_ui(self):
        ui_path = Path(__file__).resolve().parent / 'agent.ui'
        uic.loadUi(str(ui_path), self)

        self.titleLbl.setStyleSheet(f'font-size:18px;font-weight:700;color:{TEXT};')

        # ── 清空按钮 ──
        self.clearBtn.setStyleSheet(
            f"QPushButton{{background:transparent;color:{TEXT3};border:1px solid {BORDER};"
            f"border-radius:4px;padding:2px 10px;font-size:11px;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};color:{TEXT};border-color:{TEXT3};}}")
        self.clearBtn.clicked.connect(self._reset_chat)

        # ── 聊天区域 ──
        self.chatView.setStyleSheet(
            f"QListWidget{{background:{CARD};border:1px solid {BORDER};"
            f"border-radius:5px;padding:4px;}}"
            f"QListWidget::item{{border:none;}}")

        # ── 输入框 ──
        self.msgInput.setStyleSheet(
            f'QLineEdit{{background:{BG};color:{TEXT};border:1px solid {BORDER};'
            f'border-radius:4px;padding:4px 8px;font-size:12px;min-height:24px;}}'
            f'QLineEdit:focus{{border-color:{PRI};}}')
        self.msgInput.returnPressed.connect(self._on_send)

        # ── 发送按钮 ──
        self.sendBtn.setStyleSheet(
            "QPushButton{background:#07C160;color:#fff;border:none;"
            "padding:4px 16px;font-size:12px;font-weight:600;border-radius:4px;}"
            "QPushButton:hover{background:#06ad56;}"
            "QPushButton:disabled{background:#888;}")
        self.sendBtn.clicked.connect(self._on_send)

        # ── 配置按钮 ──
        self.configBtn.setStyleSheet(
            f"QPushButton{{background:{BG};color:{TEXT2};border:1px solid {BORDER};"
            f"font-size:14px;border-radius:4px;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};color:{TEXT};border-color:{PRI};}}")
        self.configBtn.clicked.connect(self._show_config)

        # ── 跨线程信号 ──
        self._reply_signal.connect(self._on_reply_from_thread)

        # ── 预加载头像 ──
        avatar_path = ROOT / 'assets' / 'YOLO.png'
        raw = QPixmap(str(avatar_path)) if avatar_path.exists() else QPixmap()
        self._avatar = raw.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not raw.isNull() else QPixmap()

    # ═══════════════════════════════════════════
    # Chat Logic
    # ═══════════════════════════════════════════

    def _reset_chat(self):
        """初始化/重置对话"""
        context = self._build_context()
        system_msg = svc.SYSTEM_PROMPT + '\n\n' + context
        self._messages = [{'role': 'system', 'content': system_msg}]
        self._loading = False
        self._set_controls_enabled(True)
        self.chatView.clear()
        self._show_welcome()

    def _build_context(self):
        """构建设备信息+项目环境上下文"""
        from main.config import load_paths
        paths = load_paths()
        gpu = f'{self.studio.gpu_name} ({self.studio.gpu_mem})' if self.studio.gpu_ok else 'CPU'
        ctx = f'当前环境：\n'
        ctx += f'GPU: {gpu}\n'
        ctx += f'类别: {", ".join(CLASSES)} ({len(CLASSES)} 类)\n'
        ctx += f'数据集: {paths.get("dataset_dir", "未配置")}\n'
        ctx += f'训练输出: {paths.get("train_output", "未配置")}\n'
        ctx += f'预测输出: {paths.get("predict_output", "未配置")}\n'
        ctx += f'导出目录: {paths.get("export_dir", "未配置")}\n'
        ctx += f'模型目录: {paths.get("models_dir", "未配置")}\n'
        # 扫描数据集目录中的 split
        ds = paths.get('dataset_dir', '')
        if ds:
            from pathlib import Path
            dp = Path(ds)
            splits = []
            for s in ('train', 'val', 'test'):
                img_dir = dp / 'images' / s
                if img_dir.exists():
                    count = len(list(img_dir.glob('*')))
                    splits.append(f'{s}:{count}张')
            if splits:
                ctx += f'数据集分片: {", ".join(splits)}\n'
        return ctx

    def _show_welcome(self):
        """在聊天区显示欢迎引导（居中展示）"""
        w = QWidget()
        w.setStyleSheet('background:transparent;')
        lo = QVBoxLayout(w)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.addStretch()
        lbl = QLabel(
            '👋 你好！我是 <b>MIRO</b>，有什么可以帮你的？<br><br>'
            '试试问我：<br>'
            '• 训练时 Loss 不下降怎么办？<br>'
            '• mAP50 和 mAP50:95 有什么区别？<br>'
            '• 过拟合了怎么调整？<br>'
            '• 不同模型怎么选（n/s/m/l/x）？<br>'
            '• 数据集怎么做数据增强？')
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f'color:#999;font-size:14px;line-height:1.8;background:transparent;')
        lbl.setAlignment(Qt.AlignCenter)
        lo.addWidget(lbl)
        lo.addStretch()

        item = QListWidgetItem(self.chatView)
        vh = self.chatView.viewport().height()
        item.setSizeHint(QSize(w.sizeHint().width(), max(w.sizeHint().height(), vh)))
        self.chatView.setItemWidget(item, w)

    def _on_send(self):
        text = self.msgInput.text().strip()
        if not text or self._loading:
            return

        self.msgInput.clear()
        self._add_bubble('user', text)
        self._messages.append({'role': 'user', 'content': text})
        self._loading = True
        self._set_controls_enabled(False)
        self._call_api()

    def _call_api(self):
        cfg = svc.load_config()
        if not cfg.get('api_key', '').strip():
            self._add_bubble('assistant', '⚠️ API Key 未配置，请先在 ⚙ 设置中配置')
            self._loading = False
            self._set_controls_enabled(True)
            return

        import threading
        def worker():
            try:
                reply = svc.send_message(list(self._messages), cfg)
                self._reply_signal.emit('reply', reply)
            except Exception as e:
                self._reply_signal.emit('error', str(e))
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _on_reply_from_thread(self, typ, text):
        """主线程槽：处理工作线程返回的结果"""
        if typ == 'reply':
            self._messages.append({'role': 'assistant', 'content': text})
            self._add_bubble('assistant', text)
        else:
            self._add_bubble('assistant', f'❌ {text}')
        self._loading = False
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        self.sendBtn.setEnabled(enabled)
        self.msgInput.setEnabled(enabled)
        if enabled:
            self.msgInput.setFocus()

    # ═══════════════════════════════════════════
    # Bubble Display (QListWidget)
    # ═══════════════════════════════════════════

    def _add_bubble(self, role: str, text: str):
        """添加一条消息气泡到聊天列表"""
        # 有欢迎页时先移除
        if self.chatView.count() == 1:
            first_item = self.chatView.item(0)
            w = self.chatView.itemWidget(first_item)
            if w and 'MIRO' in w.findChild(QLabel).text() if w.findChild(QLabel) else '':
                self.chatView.takeItem(0)

        if role == 'user':
            widget = _UserBubble(text)
        else:
            widget = _AssistantBubble(text, self._avatar)

        item = QListWidgetItem(self.chatView)
        item.setSizeHint(widget.sizeHint())
        self.chatView.setItemWidget(item, widget)
        self.chatView.scrollToBottom()

    # ═══════════════════════════════════════════
    # Config Dialog
    # ═══════════════════════════════════════════

    def _show_config(self):
        """弹出大模型接口配置对话框"""
        cfg = svc.load_config()

        dialog = QDialog(self)
        dialog.setWindowTitle('AI Agent 接口配置')
        dialog.setFixedSize(480, 200)
        dialog.setStyleSheet(f"""
            QDialog{{background:{CARD};}}
            QLabel{{color:{TEXT};font-size:13px;font-weight:500;}}
            QLineEdit,QSpinBox,QDoubleSpinBox{{
                background:{BG};color:{TEXT};border:1px solid {BORDER};
                border-radius:4px;padding:2px 8px;font-size:13px;min-height:20px;
            }}
            QLineEdit:focus{{border-color:{PRI};}}
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel('大模型接口配置')
        title.setStyleSheet(f'font-size:17px;font-weight:700;color:{TEXT};')
        layout.addWidget(title)
        layout.addSpacing(6)

        def _make_row(label_text, widget, label_width=110):
            row = QHBoxLayout()
            row.setSpacing(10)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(label_width)
            row.addWidget(lbl)
            row.addWidget(widget, 1)
            return row

        base_url = QLineEdit(cfg['base_url'])
        base_url.setPlaceholderText('https://api.openai.com/v1')
        layout.addLayout(_make_row('API 地址', base_url))

        api_key = QLineEdit(cfg['api_key'])
        api_key.setPlaceholderText('sk-xxxxxxxxxxxxxxxx')
        api_key.setEchoMode(QLineEdit.Password)
        layout.addLayout(_make_row('API Key', api_key))

        model = QLineEdit(cfg['model'])
        model.setPlaceholderText('gpt-4, deepseek, ...')
        layout.addLayout(_make_row('模型', model))

        # ── 测试连接 ──
        test_btn = QPushButton('测试连接')
        test_btn.setStyleSheet(
            f"QPushButton{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:4px;padding:7px 16px;font-size:12px;font-weight:500;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};}}"
            f"QPushButton:disabled{{color:{TEXT3};border-color:{BORDER};}}")

        def _run_test():
            """后台测试 API 连接"""
            url = base_url.text().strip()
            key = api_key.text().strip()
            mdl = model.text().strip()
            if not url or not key or not mdl:
                QMessageBox.warning(dialog, '测试连接', '请填写完整配置')
                return
            test_btn.setEnabled(False)
            test_btn.setText('测试中...')

            def _done(ok, msg=''):
                test_btn.setEnabled(True)
                test_btn.setText('测试连接')
                if ok:
                    QMessageBox.information(dialog, '测试连接', '✅ 连接成功')
                else:
                    QMessageBox.warning(dialog, '测试连接', f'❌ {msg}')

            import threading
            def _test():
                try:
                    import urllib.request, json
                    payload = json.dumps({
                        'model': mdl,
                        'messages': [{'role': 'user', 'content': 'Hi'}],
                        'max_tokens': 16,
                    }).encode()
                    req = urllib.request.Request(
                        f'{url.rstrip("/")}/chat/completions',
                        data=payload,
                        headers={'Content-Type': 'application/json',
                                 'Authorization': f'Bearer {key}'},
                        method='POST')
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        body = json.loads(resp.read().decode())
                    ok = bool(body.get('choices'))
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(0, lambda: _done(ok, '' if ok else 'API 返回异常'))
                except Exception as e:
                    from PyQt5.QtCore import QTimer
                    QTimer.singleShot(0, lambda: _done(False, str(e)[:80]))
            threading.Thread(target=_test, daemon=True).start()
        test_btn.clicked.connect(_run_test)
        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(test_btn)
        btn_row.addStretch()
        cancel_btn = QPushButton('取消')
        cancel_btn.setStyleSheet(
            f"QPushButton{{background:{BG};color:{TEXT};border:1px solid {BORDER};"
            f"border-radius:4px;padding:7px 24px;font-size:13px;font-weight:500;}}"
            f"QPushButton:hover{{background:{BTN_HOVER};}}")
        cancel_btn.clicked.connect(dialog.reject)
        save_btn = QPushButton('保存')
        save_btn.setStyleSheet(
            "QPushButton{background:#07C160;color:#fff;border:none;"
            "border-radius:4px;padding:7px 24px;font-size:13px;font-weight:600;}"
            "QPushButton:hover{background:#06ad56;}")
        save_btn.clicked.connect(lambda: self._save_config(
            dialog, base_url, api_key, model))
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        dialog.exec_()

    def _save_config(self, dialog, base_url, api_key, model):
        cfg = {
            'base_url': base_url.text().strip(),
            'api_key': api_key.text().strip(),
            'model': model.text().strip(),
        }
        svc.save_config(cfg)
        dialog.accept()
