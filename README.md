# American Football Player Detection and Video Annotation

This project provides a YOLO-based solution for detecting football players and referees in American football videos. It includes tools for video annotation, frame extraction, and dataset preparation.

## Features

- **Video Annotation**: Automatically detect and annotate football players and referees in videos
- **Real-time Processing**: Process videos with progress tracking
- **Multiple Format Support**: Supports MP4, AVI, and other common video formats
- **Customizable Models**: Use your own trained YOLO models or the provided pre-trained model
- **High-Quality Output**: Maintains original video quality with professional annotations

## Requirements

- Python 3.12+
- OpenCV
- Ultralytics YOLO
- tqdm for progress bars
- pathlib for file handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/m-abdurrehman-1/american-football.git
cd american-football
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Video Annotation

Annotate a video using the default model:

```bash
python process.py --input path/to/input_video.mp4 --output path/to/output_video.mp4
```

### Using Custom Model

Use your own trained YOLO model:

```bash
python process.py --input input_video.mp4 --output output_video.mp4 --model path/to/your_model.pt
```

### Command Line Arguments

- `--input`: Path to input video file (required)
- `--output`: Path to save the annotated video (required)
- `--model`: Path to YOLO model file (.pt) - optional, defaults to `./best.pt`

## Model Information

The default model (`best.pt`) is trained to detect:
- **Players**: Football players in different uniforms
- **Referees**: Game officials

## Output

The processed video will include:
- Bounding boxes around detected players and referees
- Confidence scores for each detection
- Color-coded annotations for different object classes

## Project Structure

```
american-football/
├── process.py          # Main video processing script
├── best.pt            # Pre-trained YOLO model
├── README.md          # This file
└── requirements.txt   # Python dependencies
               # Output directory for processed videos
```

## Performance

- Processing speed depends on video resolution and hardware
- GPU acceleration is supported through PyTorch
- Progress bar shows real-time processing status

## Troubleshooting

### Common Issues

1. **Video not opening**: Check if the video format is supported and file path is correct
2. **Model not found**: Ensure `best.pt` is in the project directory or provide correct model path
3. **Slow processing**: Consider reducing video resolution or using GPU acceleration

### Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- MKV

## License

This project is licensed under the MIT License.
