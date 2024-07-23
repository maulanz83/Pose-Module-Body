import cv2
import mediapipe as mp
import time

#bisa tidak mengunakan 3 parameter (modelComplexity, enableSegmentation, smoothSegmentation)
class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5, 
                 modelComplexity=1, enableSegmentation=False, smoothSegmentation=True):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        #self.modelComplexity = modelComplexity
        #self.enableSegmentation = enableSegmentation
        #self.smoothSegmentation = smoothSegmentation

        # Inisialisasi detektor pose dari Mediapipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode,
                                      #model_complexity=self.modelComplexity,
                                      smooth_landmarks=self.smooth,
                                      #enable_segmentation=self.enableSegmentation,
                                      #smooth_segmentation=self.smoothSegmentation,
                                      min_detection_confidence=self.detectionCon,
                                      min_tracking_confidence=self.trackCon)
        self.mp_draw = mp.solutions.drawing_utils
    
    def findPose(self, img, draw=True):
        # Proses deteksi pose
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
            # Gambar landmarks pose jika terdeteksi
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, 
                                            self.mp_pose.POSE_CONNECTIONS)
        return img
            

    def findPosition(self, img, draw=True):
        # Proses deteksi pose
        lmList = []
        # Gambar landmarks pose jika terdeteksi
        if self.results.pose_landmarks:
            #print(self.results.pose_landmarks)
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #print(id, lm)
                h, w, c = img.shape #height, width, channel
                cx, cy = int(lm.x * w), int(lm.y * h) #center x, center y
                lmList.append([id, cx, cy]) #list of landmarks
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return lmList
    



def main():
    # Ubah path video sesuai dengan yang baru
    video_path = 'C:/Users/upima/PycharmProjects/pythonProject1/venv/2-PoseEstimationBasic/PoseVideo/2.mp4'

    # Buka video
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    ptime = 0

    if not cap.isOpened():
        print(f"Error opening video file at {video_path}")
        exit()

    while True:
        # Baca frame dari video
        success, img = cap.read()
        if not success:
            print("Failed to read frame")
            break
        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)
        #untuk menetahui koodinat dari lmLIst
        if len(lmList) != 0:
            print(lmList[14])
            #menggambar pada 1 titik (14(siku))
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 225, 225), cv2.FILLED)
        # Hitung FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        # Resize frame menjadi 640x480
        img_resized = cv2.resize(img, (640, 480))
        # Tambahkan teks FPS pada frame
        cv2.putText(img_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Tampilkan frame
        cv2.imshow("Pose Estimation", img_resized)

        # Tunggu tombol 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Bebaskan sumber daya dan tutup jendela OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
