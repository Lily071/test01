"""mediapipeを使った顔向き推定

ソースはほぼ公式のコピペ
主に3箇所追加

環境(動作確認):
  Python: 3.7.9
  numpy: 1.21.4
  opencv-python: 4.3.0.38
  opencv-python-headless: 4.3.0.38
  mediapipe: 0.8.10.1

使い方:
  カメラ: python facemesh.py 0 --flip
  動画: python facemesh.py /path/to/your/video.mp4

基底ベクトルの表示意味:
  概要:
    人物がカメラ正面を向いた時、
    画面右方向をx、画面した方向をy、画面奥方向をzとして座標系を顔に固定。
    この座標系がカメラからどのように見えるかを表示。
  色との対応:
    赤線: x軸
    緑線: y軸
    青線: z軸

注意点:
  --flipした時左右反転するので、赤い軸に注意。
  軸は順番に描画しているだけなので、軸同士が重なった時前後関係がおかしく見える時がある。
  歪みに対する補正を考えてないので、画面端にいくと少しずれたように見える時がある。


参考:
  Python API(公式): https://google.github.io/mediapipe/solutions/face_mesh.html#python-solution-api
  座標点の説明(公式): https://google.github.io/mediapipe/solutions/face_mesh.html#output
  座標点の参考(Qiita): https://qiita.com/nemutas/items/6321aeca27492baeeb92
"""
#from argparse import ArgumentParser
from http.client import OK
import numpy as np
import cv2
import mediapipe as mp

sec = 1
x = [ ]
y_ud = [ ]
x_rl = [ ]

# >>> 追加1(実行時引数の取得)
# parser = ArgumentParser()
# parser.add_argument("video", type=str, help="camera ID or path to video")
# parser.add_argument("--flip", action="store_true")
# args = parser.parse_args()
# #args = parser.parse_args(args=[])

# try:
#     src = int(args.video)
# except:
#     src = args.video
# # <<< 追加1ここまで

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# # <<< 追加ここから
def face(filepass):
  src = filepass
  # # <<< 追加ここまで

  # # For webcam input:
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  cap = cv2.VideoCapture(src)

  # >>> 追加2
  #CAP_PROP_FRAME_COUNTでビデオの合計フレーム数をfpsで割ることで1秒ずつのフレームのプロットをする。
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  #mlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
  mlen =  cap.get(cv2.CAP_PROP_FRAME_COUNT)
  getsec = 1
  # fps * getsecフレーム目から取得
  cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * getsec))
  #ret, frame = cap.read()
  #print(mlen)


  # <<< 追加2ここまで
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)
          
      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

          # >>> 追加3(顔向き & 描画)

          # 使いやすくnumpyへ
          points = np.array([
            [landmark.x * width, landmark.y * height, landmark.z * width]
            for landmark in face_landmarks.landmark
          ])

          # 顔の矩形を取得
          x_min, y_min, _ = points.min(axis=0).astype(int).tolist()
          x_max, y_max, _ = points.max(axis=0).astype(int).tolist()
          x_c, y_c = (x_min + x_max) // 2, (y_min + y_max) // 2
          cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

          # 基底ベクトルを取得
          x_axis = points[356] - points[127]
          x_axis /= np.linalg.norm(x_axis)
          y_axis = points[175] - points[151]
          y_axis /= np.linalg.norm(y_axis)
          z_axis = np.cross(x_axis, y_axis)
          z_axis /= np.linalg.norm(z_axis) 

          # 基底ベクトルを描画
          scale = 200
          _x, _y = (scale * x_axis[:2] + [x_c, y_c]).astype(int).tolist()
          cv2.line(image, (x_c, y_c), (_x, _y), (0, 0, 255), 2)
          _x, _y = (scale * y_axis[:2] + [x_c, y_c]).astype(int).tolist()
          cv2.line(image, (x_c, y_c), (_x, _y), (0, 255, 0), 2)
          _x, _y = (scale * z_axis[:2] + [x_c, y_c]).astype(int).tolist()
          cv2.line(image, (x_c, y_c), (_x, _y), (255, 0, 0), 2)
          # <<< 追加3ここまで

          # <<< 追加4ここから
          #鼻のX座標
          #print("points[1]鼻のx座標,", int(points[1,0]))
          #鼻のy座標
          #print("points[1]鼻のy座標,", int(points[1,1]))
          #矩形の中心点を求める式
          #矩形の中心点から左右差を求める式 diffx
          #矩形の中心点から上下差を求める式 diffy
          (center_x, center_y) = ( (x_min + x_max)/2, (y_min + y_max)/2 )
          diffx=(center_x)-(points[1,0])
          diffy=(center_y)-(points[1,1])

                  #for i in range (0, 259):
          # フレームごとの処理時間(ms)をリストに追加
          x.append(getsec * 25) 
          x_rl.append(diffx)
          y_ud.append(diffy)        
          getsec = getsec + sec
          #print(x) #getsec秒に進んでいるフレーム数
          #print(i)
          #print(diffx)
          #print(x_rl)
          #print(y_ud)
          #print(str(mlen))
          #print(getsec)
          # <<< 追加4ここまで
        
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())

      # Flip the image horizontally for a selfie-view display.
      #cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      #cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1) if args.flip else image)

      # 'q'が入力されるまでループ
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  import module
  module.graph(x, x_rl, y_ud)
  cap.release()
  
  return 'OK'