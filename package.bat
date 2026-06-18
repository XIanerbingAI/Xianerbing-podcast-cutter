@echo off
chcp 65001 >nul
cd /d E:\ZProject\PodcastZ
echo Building Xianerbing-podcast-cutter_Mac.zip...
.venv\Scripts\python.exe _package.py
echo.
pause
