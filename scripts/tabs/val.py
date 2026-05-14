"""验证标签页"""
from scripts.tabs.base import *

class ValTab(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.validator = None
        self._build()

    def _build(self):
        lo = QHBoxLayout(self); lo.setContentsMargins(6,6,6,6); lo.setSpacing(6)

        lw = QWidget(); lw.setFixedWidth(240)
        ll = QVBoxLayout(lw); ll.setSpacing(5); ll.setContentsMargins(0,0,0,0)
        g = QGroupBox('Weights')
        gl = QVBoxLayout(g); gl.setSpacing(4); gl.setContentsMargins(8,12,8,8)
        gl.addWidget(QLabel('Model', styleSheet=f'font-size:9px;color:{TEXT2};font-weight:500;'))
        vr = QHBoxLayout()
        self.vo = QComboBox(); self.vo.setEditable(True); vr.addWidget(self.vo, 1)
        self.vo_btn = QPushButton('Browse'); self.vo_btn.clicked.connect(self._vo_browse); vr.addWidget(self.vo_btn)
        gl.addLayout(vr)
        self.bv = QPushButton('Run Validation'); self.bv.setObjectName('pri'); self.bv.clicked.connect(self._vv); gl.addWidget(self.bv)
        ll.addWidget(g)

        mc = QWidget()
        mcl = QVBoxLayout(mc); mcl.setContentsMargins(0,0,0,0); mcl.setSpacing(3)
        self.rt = QLabel('Awaiting…'); self.rt.setWordWrap(True); self.rt.setTextFormat(Qt.RichText)
        self.rt.setStyleSheet(f'color:{TEXT3};font-size:10px;padding:4px;')
        mcl.addWidget(self.rt)
        mg = QGridLayout(); mg.setSpacing(3)
        def card(lbl, attr, color):
            cw = QWidget(); cw.setStyleSheet(f'background:{BG};border-radius:5px;')
            cl = QVBoxLayout(cw); cl.setContentsMargins(6,4,6,4); cl.setSpacing(0)
            cl.addWidget(QLabel(lbl, styleSheet=f'font-size:7px;color:{TEXT3};font-weight:500;qproperty-alignment:AlignCenter;'))
            v = QLabel('—'); v.setStyleSheet(f'font-size:16px;font-weight:700;color:{color};qproperty-alignment:AlignCenter;')
            setattr(self, attr, v); cl.addWidget(v); return cw
        mg.addWidget(card('mAP@0.5','_vm50',PRI),0,0); mg.addWidget(card('mAP@0.95','_vm95',AMBER),0,1)
        mg.addWidget(card('Precision','_vp',GREEN),1,0); mg.addWidget(card('Recall','_vr',RED),1,1)
        mcl.addLayout(mg)
        self._vt = QLabel(''); self._vt.setWordWrap(True); self._vt.setTextFormat(Qt.RichText)
        self._vt.setStyleSheet(f'font-size:9px;color:{TEXT};'); self._vt.setVisible(False)
        mcl.addWidget(self._vt); mcl.addStretch()
        ll.addWidget(mc,1); lo.addWidget(lw)

        rw = QWidget(); rw.setStyleSheet(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;')
        rl = QVBoxLayout(rw); rl.setContentsMargins(6,6,6,6); rl.setSpacing(4)
        hw = QWidget(); hw.setStyleSheet('background:transparent;border:none;')
        hl = QHBoxLayout(hw); hl.setContentsMargins(2,2,2,2)
        hl.addWidget(QLabel('●', styleSheet=f'color:{PRI};font-size:7px;'))
        hl.addWidget(QLabel('Confusion Matrix', styleSheet=f'font-weight:600;font-size:10px;color:{TEXT};')); hl.addStretch()
        rl.addWidget(hw)
        cc = QWidget(); cc.setStyleSheet('background:transparent;border:none;')
        cl = QVBoxLayout(cc); cl.setContentsMargins(0,0,0,0); cl.setSpacing(0)
        self.vc = Chart(self)
        self.vc.ax.text(.5,.5,'Run validation to view',ha='center',va='center',transform=self.vc.ax.transAxes,fontsize=11,color=TEXT3)
        self.vc.ax.axis('off'); self.vc.rf()
        cl.addWidget(self.vc,1); rl.addWidget(cc,1); lo.addWidget(rw,1)
        self._rw()

    def _rw(self):
        self.vo.clear()
        for bp in sorted(ROOT.rglob('*/weights/best.pt'), key=lambda p: p.stat().st_mtime, reverse=True):
            self.vo.addItem(str(bp.relative_to(ROOT)), str(bp))

    def _vo_browse(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Select Weights', 'runs', 'PyTorch (*.pt)')
        if p:
            self.vo.setEditText(p)

    def _vv(self):
        w = self.vo.itemData(self.vo.currentIndex()) if self.vo.currentIndex()>=0 else None
        if not w or not Path(w).exists(): return
        self.bv.setEnabled(False); self.bv.setText('Running…')
        gpu = self.studio.gpu_ok
        class V(QThread):
            d = pyqtSignal(object)
            def run(self):
                try:
                    from ultralytics import YOLO
                    v = cfg['validation']
                    r = YOLO(str(w)).val(data=DATA_YAML, device='0' if gpu else 'cpu',
                        imgsz=v['imgsz'], batch=v['batch'], conf=v['conf'], iou=v['iou'],
                        plots=True, verbose=False)
                    self.d.emit(r)
                except Exception as e:
                    import traceback; traceback.print_exc(); self.d.emit(e)
        self.validator = V(); self.validator.d.connect(self._vd); self.validator.start()

    def _vd(self, r):
        self.bv.setEnabled(True); self.bv.setText('Run Validation'); self.validator = None
        if isinstance(r, Exception):
            import traceback
            self.rt.setText(f'❌ {r}\n{traceback.format_exc()}'); self._vt.setVisible(False)
            QMessageBox.critical(self, 'Validation Error', f'Validation failed:\n{str(r)}'); return
        try:
            b = r.box
            self._vm50.setText(f'{b.map50:.4f}'); self._vm95.setText(f'{b.map:.4f}')
            self._vp.setText(f'{b.p.mean():.4f}'); self._vr.setText(f'{b.r.mean():.4f}')
            rows = ''
            for i, nm in enumerate(CLASSES):
                mi = b.maps50[i] if hasattr(b,'maps50') and len(b.maps50)>i else 0
                pi = b.p[i] if hasattr(b,'p') and len(b.p)>i else 0
                ri = b.r[i] if hasattr(b,'r') and len(b.r)>i else 0
                rows += f'<tr><td style="padding:1px 6px">{nm}</td><td style="padding:1px 6px;color:{PRI}">{mi:.4f}</td><td style="padding:1px 6px">{pi:.4f}</td><td style="padding:1px 6px">{ri:.4f}</td></tr>'
            self._vt.setText(f'<table style="width:100%;font-size:10px;border-collapse:collapse;"><tr style="color:{TEXT3}"><th>Class</th><th>mAP</th><th>P</th><th>R</th></tr>{rows}</table>')
            self._vt.setVisible(True); self.rt.setText('')
            sd = Path(r.save_dir) if hasattr(r,'save_dir') else None
            if sd:
                imgs = list(sd.glob('confusion_matrix*.png'))
                if imgs:
                    self.vc.ax.clear(); self.vc.ax.axis('on')
                    self.vc.ax.imshow(plt.imread(str(imgs[0]))); self.vc.ax.axis('off'); self.vc.rf()
        except Exception as e:
            import traceback; traceback.print_exc()
            self.rt.setText(f'❌ {str(e)}\n{traceback.format_exc()}')
            QMessageBox.critical(self, 'Display Error', f'Display failed:\n{str(e)}')
