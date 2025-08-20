# Face Recognition Rapper Look-Alike
## Overview
Real-time face recognition application that detects faces in a webcam, matches them to a dataset of celebrity images, and tells you who your rapper look-alike is
Uses:
  ● face_recognition for recognition
  ● OpenCV for video capture and drawing 
  ● beautifulsoup + requests for web scraping
  ● NumPy for numerical tasks

## Installation
Clone repo & then
install dependencies in requirements.txt
Note: face_recognition depends on dlib, which requires CMake; installing face_recognition without them may fail

## Usage
python3 scrape_images.py
Scrapes images of rappers and saves them in Images/
python3 encode_faces.py
Generates the files in data/
python3 recognition.py
Run the recognition webcam, press Enter to quit

## Improvements for the future
Better photo choices for some rappers
Reducing overprediction of Lil Pump 
