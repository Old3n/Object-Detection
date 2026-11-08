from PyInstaller.utils.hooks import collect_all, copy_metadata

ultra_data, ultra_bins, ultra_hidden = collect_all('ultralytics')
a = Analysis(
    ['AutoPilot.py'],
    pathex=[],
    binaries=ultra_bins,
    datas=[('web', 'web'), ('data', 'data'), ('yolo26n.pt', '.')] + ultra_data + copy_metadata('ultralytics'),
    hiddenimports=ultra_hidden + [
        'RiskVideoBenchmark', 'VideoBenchmark', 'SurroundCameraBenchmark',
        'ModelBenchmark', 'EvaluateModel', 'train_pedestrian_model',
        'tools.list_cameras', 'tools.prepare_jaad_dataset', 'tools.kitti_to_video',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'IPython'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='AutoPilot', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=True,
)
