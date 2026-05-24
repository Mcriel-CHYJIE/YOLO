# =============================================================================
# YOLO Training Studio — AI Agent 标签页
# 学习交流许可 | Copyright (c) 2025 Mcriel-CHYJIE | 禁止商用倒卖
# =============================================================================

"""AI Agent 标签页 — YOLO 专家助手聊天"""

from pathlib import Path
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QTextDocument
from PyQt5.QtCore import QUrl, QSize
from main.core.base import *
from . import service as svc


class _ApiWorker(QObject):
    """后台调用 API 的工作线程对象"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, messages, config):
        super().__init__()
        self.messages = messages
        self.config = config

    def run(self):
        try:
            reply = svc.send_message(self.messages, self.config)
            self.finished.emit(reply)
        except Exception as e:
            self.error.emit(str(e))


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
        lbl.setMaximumWidth(560)
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
            ava = QLabel()
            ava.setPixmap(avatar_pixmap)
            ava.setFixedSize(40, 40)
            ava.setStyleSheet('border-radius:6px;')
            lo.addWidget(ava)
            lo.addSpacing(8)
        else:
            lo.addSpacing(48)

        # 气泡
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(520)
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

        # ── 预加载头像 ──
        avatar_path = ROOT / 'assets' / 'YOLO.png'
        raw = QPixmap(str(avatar_path)) if avatar_path.exists() else QPixmap()
        self._avatar = raw.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not raw.isNull() else QPixmap()

    # ═══════════════════════════════════════════
    # Chat Logic
    # ═══════════════════════════════════════════

    def _reset_chat(self):
        """初始化/重置对话"""
        self._messages = [{'role': 'system', 'content': svc.SYSTEM_PROMPT}]
        self._loading = False
        self._set_controls_enabled(True)
        self.chatView.clear()
        self._show_welcome()

    def _show_welcome(self):
        """在聊天区显示欢迎引导"""
        w = QWidget()
        w.setStyleSheet('background:transparent;')
        lo = QVBoxLayout(w)
        lo.setAlignment(Qt.AlignCenter)
        lbl = QLabel(
            '👋 你好！我是 <b>YOLO 助手</b>，有什么可以帮你的？<br><br>'
            '试试问我：<br>'
            '• YOLOv8 训练参数怎么调？<br>'
            '• 什么是 mAP？<br>'
            '• 小模型还是大模型？<br>'
            '• 数据不够怎么办？')
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f'color:#999;font-size:14px;line-height:1.8;background:transparent;')
        lbl.setAlignment(Qt.AlignLeft)
        lo.addWidget(lbl)

        item = QListWidgetItem(self.chatView)
        item.setSizeHint(w.sizeHint())
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
            self._on_api_error('⚠️ API Key 未配置，请先在 ⚙ 设置中配置')
            return

        self._thread = QThread(self)
        worker = _ApiWorker(list(self._messages), cfg)
        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)
        worker.finished.connect(self._on_api_reply)
        worker.error.connect(self._on_api_error)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._on_api_done)
        self._thread.start()

    def _on_api_reply(self, reply):
        self._messages.append({'role': 'assistant', 'content': reply})
        self._add_bubble('assistant', reply)

    def _on_api_error(self, err):
        self._add_bubble('assistant', f'❌ {err}')

    def _on_api_done(self):
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
            if w and 'YOLO 助手' in w.findChild(QLabel).text() if w.findChild(QLabel) else '':
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
        dialog.setFixedSize(520, 300)
        dialog.setStyleSheet(f"""
            QDialog{{background:{CARD};}}
            QLabel{{color:{TEXT};font-size:13px;font-weight:500;}}
            QLineEdit,QSpinBox,QDoubleSpinBox{{
                background:{BG};color:{TEXT};border:1px solid {BORDER};
                border-radius:4px;padding:6px 10px;font-size:13px;min-height:28px;
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

        row3 = QHBoxLayout()
        row3.setSpacing(10)
        model = QLineEdit(cfg['model'])
        model.setPlaceholderText('gpt-4, deepseek, ...')
        row3.addWidget(QLabel('模型'), 0)
        row3.addWidget(model, 1)
        temp = QDoubleSpinBox()
        temp.setRange(0.0, 2.0)
        temp.setSingleStep(0.1)
        temp.setValue(cfg.get('temperature', 0.7))
        row3.addWidget(QLabel('温度'), 0)
        row3.addWidget(temp, 1)
        max_tokens = QSpinBox()
        max_tokens.setRange(64, 65536)
        max_tokens.setSingleStep(512)
        max_tokens.setValue(cfg.get('max_tokens', 4096))
        row3.addWidget(QLabel('Token'), 0)
        row3.addWidget(max_tokens, 1)
        layout.addLayout(row3)
        layout.addStretch()

        btn_row = QHBoxLayout()
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
            dialog, base_url, api_key, model, temp, max_tokens))
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        dialog.exec_()

    def _save_config(self, dialog, base_url, api_key, model, temp, max_tokens):
        cfg = {
            'base_url': base_url.text().strip(),
            'api_key': api_key.text().strip(),
            'model': model.text().strip(),
            'temperature': temp.value(),
            'max_tokens': max_tokens.value(),
        }
        svc.save_config(cfg)
        dialog.accept()
