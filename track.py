import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def get_fourcc(output_path: Path):
    ext = output_path.suffix.lower()
    if ext == '.mp4':
        return cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        return cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError(f"Unsupported video format: {ext}")


def annotate_video(input_path: Path, output_path: Path, model_path: Path):
    model = YOLO(str(model_path))
    class_names = model.names

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = get_fourcc(output_path)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    with tqdm(total=total_frames, desc="tracking", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False)[0]

            annotator = Annotator(
                frame,
                line_width=2,
                font_size=6,
                font='Arial.ttf',
                pil=True,
                example=class_names
            )

            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy[0].tolist()
                label = f"{class_names[cls]} {conf:.2f}"

                # Add tracking ID if available
                if box.id is not None:
                    track_id = int(box.id.item())
                    label += f" ID:{track_id}"

                color = colors(cls, bgr=True)
                annotator.box_label(xyxy, label, color=color)

            out.write(annotator.result())
            pbar.update(1)

    cap.release()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate video using YOLO model")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save annotated video")
    parser.add_argument("--model", help="Path to YOLO model file (.pt), optional")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model_path = Path(args.model).resolve() if args.model else Path("./models/best.pt")

    annotate_video(input_path, output_path, model_path)
