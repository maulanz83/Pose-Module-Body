import cv2
import mediapipe as mp
import time

# Ubah pathq video sesuai dengan yang baru

video_path = 'C:/Users/upima/PycharmProjects/pythonProject1/venv/2-PoseEstimationBasic/PoseVideo/3.mp4'

# Buka video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file at {video_path}")
    exit()

# Inisialisasi detektor pose dari Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

ptime = 0

while True:
    # Baca frame dari video
    success, img = cap.read()
    if not success:
        print("Failed to read frame")
        break

    # Proses deteksi pose
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # Gambar landmarks pose jika terdeteksi
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

    # Resize frame menjadi 640x480
    img_resized = cv2.resize(img, (640, 480))

    # Hitung FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    # Tambahkan teks FPS pada frame
    cv2.putText(img_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Tampilkan frame
    cv2.imshow("Pose Estimation", img_resized)

    # Tunggu tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya dan tutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()
