import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--windowed',
    '--hidden-import=tkinter',
    '--hidden-import=PIL._tkinter_finder',
])