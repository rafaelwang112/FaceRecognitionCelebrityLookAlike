import cv2
import numpy as np
import face_recognition
import os
import json

path = 'data'
img_path = 'Images'
cache = {}
thumbnail_width = 400

encodings_path = os.path.join(path, "encodings.npy")
names_path = os.path.join(path, "names.json")

if not os.path.exists(encodings_path) or not os.path.exists(names_path):
    print ("Run encode_faces.py first.")
    exit()

my_encode_list = np.load(encodings_path)
with open (names_path, "r") as file:
    names = json.load(file)

capture = cv2.VideoCapture(1)
if not capture.isOpened():
    print ("Could not open camera.")
    exit()
    
thresh = 0.55
list_thresh = 0.6
cached_for = 45
alpha = 0.05
consensus = 3
track_list = []
frame_cnt = 0

while True:
    suc, img = capture.read()
    img_shrink = cv2.resize(img,(0,0),None,0.20,0.20)
    img_shrink = cv2.cvtColor(img_shrink, cv2.COLOR_BGR2RGB)

    #for multiple faces in frame
    faces_loc = face_recognition.face_locations(img_shrink)
    encode_imgs = face_recognition.face_encodings(img_shrink, faces_loc, num_jitters=2, model = "small")

    frame_cnt +=1
    #delete expired tracks
    if track_list:
        i = 0
        while i<len(track_list):
            if (frame_cnt -track_list[i]['last_seen'])> cached_for:
                del track_list[i]
            else:
                i+=1

    #find matches
    for encoded_face, face_loc in zip(encode_imgs, faces_loc):
        name = None
        index = -1
        if (len(track_list)>0):
            dists = []
            for track in track_list:
                d = encoded_face - track['encoding']
                dist = np.linalg.norm(d)
                dists.append(dist)
            index = int (np.argmin(dists))
            if dists[index]>=thresh:
                index = -1
        
        if index>=0:
            track_list[index]['encoding'] = (1-alpha)*track_list[index]['encoding'] + alpha*encoded_face
            track_list[index]['last_seen'] = frame_cnt
            if track_list[index]['name'] is None:
                list_dist = face_recognition.face_distance(my_encode_list, encoded_face)
                best_index = np.argmin(list_dist) 
                if list_dist[best_index]<list_thresh:
                    candidate = names[best_index]
                    if candidate in track_list[index]['votes']:
                        track_list[index]['votes'][candidate] +=1
                    else: #for if another candidate appears
                        track_list[index]['votes'][candidate]=1
                    if track_list[index]['votes'][candidate] >=consensus:
                        track_list[index]['name'] = candidate
            name = track_list[index]['name']

        else:
            list_dist = face_recognition.face_distance(my_encode_list, encoded_face) 
            best_index = np.argmin(list_dist) 
            new_track = {'encoding':encoded_face.copy(), 'name': None, 'last_seen':frame_cnt, 'votes': {}}
            if list_dist[best_index]<list_thresh:
                    candidate = names[best_index]
                    new_track['votes'][candidate]=1
                    if new_track['votes'][candidate] >=consensus:
                        new_track['name'] = candidate

            track_list.append(new_track)
            name = new_track['name']

        y1,x2,y2,x1 = face_loc
        y1,x2,y2,x1 = y1*5, x2*5, y2*5, x1*5
        cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 2)
        if name is not None:
            thumbnail = None
            if name in cache:
                thumbnail = cache[name]
            else:
                jpg_path = os.path.join(img_path, name + '.jpg')
                src = cv2.imread(jpg_path)
                h = src.shape[0]
                w = src.shape[1]
                new_w = thumbnail_width
                new_h = int(h*(new_w/float(w)))
                thumbnail = cv2.resize(src, (new_w, new_h))
                cache[name] = thumbnail
            
            thumbnail_h = thumbnail.shape[0]
            thumbnail_w = thumbnail.shape[1]
            box_center = (x1+x2) //2
            left_edge = box_center - (thumbnail_w//2)

            if left_edge < 0:
                left_edge = 0
            if left_edge + thumbnail_w > img.shape[1]:
                left_edge = img.shape[1] - thumbnail_w

            right_edge = left_edge + thumbnail_w
            bottom_edge = y1 - 5
            top_edge = bottom_edge - thumbnail_h

            if top_edge <0:
                top_edge = 0
                bottom_edge = thumbnail_h

            img[top_edge:bottom_edge, left_edge:right_edge] = thumbnail
            cv2.rectangle(img, (left_edge, top_edge), (right_edge, bottom_edge), (255, 0, 0), 1)
            
            (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 5, 2)
            text_x = left_edge + (thumbnail_w - text_w) // 2
            text_y = top_edge - 5
            if text_y - text_h <0:
                text_y = text_h + 2

            cv2.putText(img, name, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 5)
            cv2.putText(img, name, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 2)
            
    cv2.imshow('Webcam', img)
    if (cv2.waitKey(1) & 0xFF == 13):
        break

capture.release()