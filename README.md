# Face Recognition Rapper Look-Alike
Passion project that matches your face to your favorite rappers
## Overview
Real-time face recognition application that detects faces in a webcam, matches them to a dataset of celebrity images, and tells you who your rapper look-alike is

Uses:
  1. face_recognition for recognition  
  2. OpenCV for video capture and drawing  
  3. beautifulsoup + requests for web scraping  
  4. NumPy for numerical tasks

## Installation
Clone repo & then install dependencies in requirements.txt  
Note: face_recognition depends on dlib, which requires CMake; installing face_recognition without them may fail

## Usage
To scrape images of rappers and saves them in Images directory:
```bash
python3 scrape_images.py
```  
To generate the files in data directory:
```bash
python3 encode_faces.py
```  
To run the recognition webcam (press Enter to quit):
```bash
python3 recognition.py
```  
## Improvements for the Future
Better photo choices for some rappers  
Reducing overprediction of Lil Pump 
