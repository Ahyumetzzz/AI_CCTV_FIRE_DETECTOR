from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

import cv2
import supervision as sv
import torch
import onnxruntime as ort
import time
import os

#Konfigurasi ketersediaan CUDA dan ONNX Runtime
print("CUDA tersedia (PyTorch):", torch.cuda.is_available())
print("ONNX Runtime Providers:", ort.get_available_providers())

ROBOFLOW_API_KEY = "RyvHfttQOXdV4ijDSDqm"

#Inisialisasi Annotator
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

#Variabel Kontrol Pemutaran
PLAYBACK_SPEED = 1.0
FRAME_SKIP_COUNT = 0
CURRENT_FRAME_INDEX = 0
VIDEO_FPS = 30.0

#Konfigurasi penyimpanan hasil deteksi
OUTPUT_FOLDER = r"C:\ModeKerja\PROTOTIPE2_YOLO_FIRE_DETECTOR\inference_results\model_fire-detection-9mwyw14_'BREAKING NEWS ! Detik-detik Tangki Kilang Minyak Pertamina di Cilacap Terbak.mp4'" # Nama folder untuk menyimpan gambar
os.makedirs(OUTPUT_FOLDER, exist_ok=True) # Buat folder jika belum ada

#FPS Video dari OpenCV (biar lebih robust)
temp_cap = cv2.VideoCapture(r"C:\ModeKerja\PROTOTIPE2_YOLO_FIRE_DETECTOR\model_training\dataset\raw\BREAKING NEWS ! Detik-detik Tangki Kilang Minyak Pertamina di Cilacap Terbak.mp4")
if temp_cap.isOpened():
    ret_fps = temp_cap.get(cv2.CAP_PROP_FPS)
    if ret_fps > 0:
        VIDEO_FPS = ret_fps
    temp_cap.release()
print(f"FPS Video Terdeteksi (dari OpenCV): {VIDEO_FPS}")

#Label, bounding box, dan mask
def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    global PLAYBACK_SPEED, FRAME_SKIP_COUNT, CURRENT_FRAME_INDEX, VIDEO_FPS

    CURRENT_FRAME_INDEX = video_frame.frame_id if hasattr(video_frame, 'frame_id') and video_frame.frame_id is not None else CURRENT_FRAME_INDEX + 1

    if FRAME_SKIP_COUNT > 0:
        if CURRENT_FRAME_INDEX % (FRAME_SKIP_COUNT + 1) != 0:
            return
        else:
            FRAME_SKIP_COUNT = 0

    detections = sv.Detections.from_inference(predictions)

    annotated_image = video_frame.image.copy()

    #Kalau ada anomali, nanti bisa otomatis simpan frame
    has_detections = "predictions" in predictions and predictions["predictions"]

    if has_detections:
        labels = [
            f"{p.get('class', 'unknown')} {p.get('confidence', 0.0):.2f}"
            for p in predictions["predictions"]
        ]

        annotated_image = mask_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
    else:
        cv2.putText(annotated_image, "No Detections",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


    #Informasi kecepatan dan frame ke tampilan
    cv2.putText(annotated_image, f"Speed: {PLAYBACK_SPEED:.1f}x | Frame: {CURRENT_FRAME_INDEX}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #Simpan Hasil Deteksi ke PNG
    #Nama file output
    output_filename = os.path.join(OUTPUT_FOLDER, f"frame_{CURRENT_FRAME_INDEX:05d}.png")
    cv2.imwrite(output_filename, annotated_image)
    print(f"Frame {CURRENT_FRAME_INDEX} disimpan sebagai {output_filename}")

    cv2.imshow("Predictions with Segmentation", annotated_image)

    #Rumus kecepatan video
    delay = 1
    if VIDEO_FPS > 0 and PLAYBACK_SPEED > 0:
        delay = int(1000 / (VIDEO_FPS * PLAYBACK_SPEED))
        if delay <= 0:
            delay = 1
    key = cv2.waitKey(delay) & 0xFF

    #Kontrol Keyboard
    if key == ord('q'):
        print("Menghentikan pipeline...")
        pipeline.terminate()
        cv2.destroyAllWindows()
    elif key == ord('='):
        PLAYBACK_SPEED = min(5.0, PLAYBACK_SPEED + 0.5)
        print(f"Kecepatan pemutaran: {PLAYBACK_SPEED:.1f}x")
    elif key == ord('-'):
        PLAYBACK_SPEED = max(0.1, PLAYBACK_SPEED - 0.5)
        print(f"Kecepatan pemutaran: {PLAYBACK_SPEED:.1f}x")
    elif key == ord('f'):
        FRAME_SKIP_COUNT = 99
        print(f"Maju 100 frame dari {CURRENT_FRAME_INDEX}...")
    elif key == ord('r'):
        PLAYBACK_SPEED = 1.0
        FRAME_SKIP_COUNT = 0
        print("Kecepatan direset ke 1.0x")


#Inisialisasi Pipeline 
pipeline = InferencePipeline.init(
    model_id="fire-detection-9mwyw/14",
    api_key=ROBOFLOW_API_KEY,
    video_reference=r"C:\ModeKerja\PROTOTIPE2_YOLO_FIRE_DETECTOR\model_training\dataset\raw\BREAKING NEWS ! Detik-detik Tangki Kilang Minyak Pertamina di Cilacap Terbak.mp4",
    on_prediction=my_custom_sink,
    #start_frame_index=0, (untuk mulai dari frame tertentu)
    #stop_frame_index=None, (untuk stop pada frame tertentu)
)

pipeline.start()
pipeline.join()