from ultralytics import YOLO
import cv2
import supervision as sv
import torch
import onnxruntime as ort
import os

#Konfigurasi path model, path video yang mau diinferensi (kalau mau pakai webcam ganti 0), dan folder output hasil deteksi
LOCAL_MODEL_PATH = r"C:\ModeKerja\PROTOTIPE2_YOLO_FIRE_DETECTOR\models\yolov8s\best1.pt" 
VIDEO_PATH = r"C:\ModeKerja\PROTOTIPE2_YOLO_FIRE_DETECTOR\model_training\dataset\raw\videoplayback.mp4"
OUTPUT_FOLDER = r"C:\ModeKerja\PROTOTIPE2_YOLO_FIRE_DETECTOR\inference_results\model_fire-detection-9mwyw14_'videoplayback.mp4'"

#Untuk mastiin folder output ada. Kalau gak ada otomatis bikin baru
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#Meriksa CUDA dan ONNX Runtime
print("CUDA tersedia (PyTorch):", torch.cuda.is_available())
#Kalau 'CUDAExecutionProvider', berarti ONNX Runtime bisa pakai GPU CUDA lokal
print("ONNX Runtime Providers:", ort.get_available_providers())

#Memuat model YOLO dari path lokal
try:
    model = YOLO(LOCAL_MODEL_PATH)
    print(f"Model lokal {LOCAL_MODEL_PATH} berhasil dimuat.")
    #Model secara otomatis bakal tau nama kelasnya dari hasil training
    CLASS_NAMES_DICT = model.names
except Exception as e:
    print(f"Error saat memuat model lokal: {e}")
    print("Pastikan path model benar dan file model (.pt) tidak rusak.")
    exit()

#Inisialisasi Annotator
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

#Buka video dan dapatin info FPS
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Tidak dapat membuka video dari {VIDEO_PATH}")
    exit()
VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
if VIDEO_FPS <= 0:
    VIDEO_FPS = 30.0 #Fallback jika FPS tidak terbaca
print(f"FPS Video Terdeteksi: {VIDEO_FPS}")

#Loop inferensi frame per-frame
CURRENT_FRAME_INDEX = 0
PLAYBACK_SPEED = 1.0 #Kontrol kecepatan
FRAME_SKIP_COUNT = 0 #Untuk fitur maju frame

while cap.isOpened():
    #Untuk mempercepat video
    if FRAME_SKIP_COUNT > 0:
        for _ in range(FRAME_SKIP_COUNT):
            ret, frame = cap.read()
            if not ret:
                break 
            CURRENT_FRAME_INDEX += 1
        FRAME_SKIP_COUNT = 0

    ret, frame = cap.read()
    if not ret:
        print("Video selesai.")
        break

    CURRENT_FRAME_INDEX += 1

    #Lakukan Inferensi
    device='cuda' 
    'verbose=False' 
    results = model(frame, device='cuda', verbose=False)

    #Proses Hasil dengan Supervision
    detections = sv.Detections.from_ultralytics(results[0])

    #Pastikan ada deteksi sebelum mencoba mengakses class_id
    if detections.xyxy.shape[0] > 0:
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
    else:
        labels = [] #Kalau gak ada deteksi, gak ada label

    #Anotasi Gambar
    annotated_image = frame.copy() # Mulai dengan frame asli

    #Anotasi masker, box, dan label
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    #Tampilan info kecepatan dan frame
    cv2.putText(annotated_image, f"Speed: {PLAYBACK_SPEED:.1f}x | Frame: {CURRENT_FRAME_INDEX}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #Simpan Hasil Deteksi ke PNG
    output_filename = os.path.join(OUTPUT_FOLDER, f"frame_{CURRENT_FRAME_INDEX:05d}.png")
    cv2.imwrite(output_filename, annotated_image)

    #Nama windows yang muncul
    cv2.imshow("Fire Detector", annotated_image)

    #Kontrol Kecepatan dan Keyboard
    delay = 1
    if VIDEO_FPS > 0 and PLAYBACK_SPEED > 0:
        delay = int(1000 / (VIDEO_FPS * PLAYBACK_SPEED))
        if delay <= 0:
            delay = 1
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('q'): # 'q' untuk keluar
        print("Menghentikan pemutaran...")
        break
    elif key == ord('='): # '=' untuk mempercepat
        PLAYBACK_SPEED = min(5.0, PLAYBACK_SPEED + 0.5)
        print(f"Kecepatan pemutaran: {PLAYBACK_SPEED:.1f}x")
    elif key == ord('-'): # '-' untuk memperlambat
        PLAYBACK_SPEED = max(0.1, PLAYBACK_SPEED - 0.5)
        print(f"Kecepatan pemutaran: {PLAYBACK_SPEED:.1f}x")
    elif key == ord('f'): # 'f' untuk maju 100 frame
        FRAME_SKIP_COUNT = 99
        print(f"Maju 100 frame dari {CURRENT_FRAME_INDEX}...")
    elif key == ord('b'): # 'b' untuk mundur 100 frame
        # Untuk mundur, kita harus mengatur ulang posisi video
        new_frame_pos = max(0, CURRENT_FRAME_INDEX - 100 - 1) # -1 because it will increment +1 immediately
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
        CURRENT_FRAME_INDEX = new_frame_pos
        print(f"Mundur ke frame {CURRENT_FRAME_INDEX}...")
    elif key == ord('r'): # 'r' untuk reset kecepatan
        PLAYBACK_SPEED = 1.0
        FRAME_SKIP_COUNT = 0
        print("Kecepatan direset ke 1.0x")

cap.release()
cv2.destroyAllWindows()