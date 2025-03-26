# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\johan\\Documents\\programming\\git repos\\inf_MLP\\exeMLP3d.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\johan\\Documents\\programming\\git repos\\inf_MLP\\venv\\Lib\\site-packages\\vispy\\glsl', 'vispy\\glsl'), ('C:\\Users\\johan\\Documents\\programming\\git repos\\inf_MLP\\venv\\Lib\\site-packages\\vispy\\io\\_data\\spatial-filters.npy', 'vispy/io/_data'), ('C:\\Users\\johan\\Documents\\programming\\git repos\\inf_MLP\\venv\\Lib\\site-packages\\vispy\\io\\_data\\fonts', 'vispy/io/_data/fonts')],
    hiddenimports=['vispy.app.backends._pyqt5'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='exeMLP3d',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
