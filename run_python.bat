@echo off
cd dish-segmentor\venv\Scripts
call activate
cd ..\..\
python dishy.py --path PASTE_YOUR_IMAGE_PATH_HERE
