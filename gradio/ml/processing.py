import cv2
import ocr_module, cv2_converter

from ultralytics import YOLO
from PIL import Image
import torch.nn.functional as F

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

import supervision as sv
import csv

model = YOLO('./best.pt')

# ----------------------------- start-up configuration -----------------------------

def startup_conf():
    global processor, ocr_model
    processor = TrOCRProcessor.from_pretrained('./processor')
    ocr_model = VisionEncoderDecoderModel.from_pretrained('./tr_ocr_m')

# ----------------------------- image processing -----------------------------

def recognize(base_path: str):
    image = Image.open(base_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(
        pixel_values,
        output_scores=True,
        return_dict_in_generate=True
    )
    ids, scores = generated_ids['sequences'], generated_ids['scores']
    generated_text = processor.batch_decode(ids, skip_special_tokens=True)[0]  # лейбл
    _, probabilities = ocr_module.generate_proba(scores, ids, processor)  # вероятности 
    is_correct = ocr_module.is_proper(generated_text)
    return probabilities, generated_text, is_correct

def sign_detection(source: str, result_name: str, annotated_mode: bool=False) -> None:
    yolo_predict = model(source)[0]
    if annotated_mode:
        with open(source.split('.')[0] + '.txt', mode='w') as label_file:
            for bbox in yolo_predict.boxes:
                class_bbox = str(int(bbox.cls.cpu().item()))
                coordinates_bbox = ' '.join([str(el) for el in bbox.xywhn.cpu().tolist()[0]])
                label_file.write(class_bbox + ' ' + coordinates_bbox + '\n')
    detections = sv.Detections.from_ultralytics(yolo_predict)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    annotated_image = cv2.imread(source)
    annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    cv2.imwrite(result_name, annotated_image)

def sign_recognition(source: str, result_name: str) -> None:
    cropped_name = './files/cropped_image_' + result_name
    results = model.predict(source)
    cropped_image = cv2_converter.crop(source, results)
    cv2.imwrite(cropped_name, cropped_image)
    probability, number, is_correct = recognize(cropped_name)
    print(number, probability)
    return number, probability
    

# ----------------------------- video processing -----------------------------

# video process function
def process_video(source: str, destination: str = 'result.mp4') -> None:
    video_info = sv.VideoInfo.from_video_path(video_path=source)
    frames_generator = sv.get_video_frames_generator(source_path=source)
    box_annotator = sv.BoundingBoxAnnotator()

    start_interval = None
    interval_ids = set()

    with sv.VideoSink(target_path=destination, video_info=video_info, codec='h264') as sink:
        for i, frame in enumerate(frames_generator):
            print(i)
            result = model.track(frame, verbose=False, persist=True, agnostic_nms=True)[0]
            if len(result.boxes) and result.boxes.id and start_interval is None:
                print(result.boxes)
                start_interval = int(i / video_info.fps)
                interval_ids.update(result.boxes.id.cpu().tolist())
            elif start_interval:
                print('yes')
                save_interval(start_interval, int(i / video_info.fps), len(interval_ids))
                start_interval = None
                interval_ids = set()
            if len(result.boxes) and result.boxes.id:
                interval_ids.update(result.boxes.id.cpu().tolist())
            detections = sv.Detections.from_ultralytics(result)
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections)
            sink.write_frame(frame=annotated_frame)
        if len(interval_ids):
            save_interval(start_interval, int(video_info.total_frames / video_info.fps), len(interval_ids))

def save_interval(start: int, end: int, overall: int) -> None:
    with open('detections.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([start, end, overall])


if __name__ == '__main__':
    process_video('cart.mp4')
    
    