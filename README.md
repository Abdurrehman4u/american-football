# American Football Player Detection and Video Annotation

This project provides advanced solutions for detecting, tracking, and analyzing American football players, referees, and the ball in videos. It includes tools for video annotation, player tracking, and bird's-eye view visualization.

## Features

- **Video Annotation**: Automatically detect and annotate football players, referees, and the ball
- **Player Tracking**: Advanced player tracking with persistent IDs across frames
- **Bird's-Eye View**: Real-time overhead visualization of player positions
- **Multiple Format Support**: Supports MP4 and AVI video formats
- **Real-time Processing**: Process videos with progress tracking
- **Data Export**: Save tracking data in JSON format for further analysis

## Requirements

- Python 3.12+
- OpenCV
- Ultralytics YOLO
- PyTorch (for GPU acceleration)
- tqdm for progress bars
- pathlib for file handling
- scipy for tracking optimizations
- sklearn for similarity metrics

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

3. Place required model files in the `models` directory:
- `american_v2.pt`: Main detection model
- `field_seg.pt`: Field segmentation model
- `field.jpg`: Bird's-eye view template image

## Usage

### Basic Video Processing

```bash
python process.py --input path/to/input_video.mp4 --output path/to/output_video.mp4
```

### Advanced Player Tracking with Bird's-Eye View

```bash
python birdeye.py --input path/to/input_video.mp4 --output path/to/output_video.mp4
```

### Command Line Arguments

- `--input`: Path to input video file (required)
- `--output`: Path to save the processed video (required)
- `--model`: Path to YOLO model file (.pt) - optional, defaults to `./models/american_v2.pt`

## Project Structure

```
american-football/
├── process.py          # Basic video processing script
├── track.py           # Advanced player tracking script
├── birdeye.py        # Bird's-eye view visualization
├── models/           # Directory for model files
│   ├── american_v2.pt    # Main detection model
│   ├── field_seg.pt      # Field segmentation model
│   └── field.jpg         # Bird's-eye template
├── README.md         # Documentation
└── requirements.txt  # Python dependencies
```

## Features in Detail

### Video Processing (process.py)
- Basic player and referee detection
- Bounding box annotations
- Confidence score display

### Player Tracking (track.py)
- Advanced multi-object tracking
- Persistent player IDs
- Velocity vectors visualization
- JSON export of tracking data

### Bird's-Eye View (birdeye.py)
- Real-time overhead visualization
- Player position mapping
- Field segmentation
- Interactive overlay with player IDs


## Troubleshooting

### Common Issues

1. **Video not opening**: Check if the video format is supported and file path is correct
2. **Model not found**: Ensure all required model files are in the `models` directory
3. **Slow processing**: Consider reducing video resolution or using GPU acceleration
4. **Bird's-eye view not showing**: Verify `field.jpg` exists in the models directory

### Supported Video Formats

- MP4 (recommended)
- AVI

## License

This project is licensed under the MIT License.
