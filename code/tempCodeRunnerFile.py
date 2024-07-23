# Ubah path video sesuai dengan yang baru
    video_path = 'C:/Users/upima/PycharmProjects/pythonProject1/venv/2-PoseEstimationBasic/PoseVideo/1.mp4'

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