import cv2
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from collections import defaultdict

def get_fourcc(output_path: Path):
    """Determines the FourCC code for the output video file based on its extension."""
    ext = output_path.suffix.lower()
    if ext == '.mp4':
        return cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        return cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError(f"Unsupported video format: {ext}")

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_center(xyxy):
    """Calculates the center of a bounding box."""
    x1, y1, x2, y2 = xyxy
    # FIX: Explicitly convert Tensors to standard Python floats for JSON compatibility
    return [float((x1 + x2) / 2), float((y1 + y2) / 2)]

def get_avg_color(image, center, size=10):
    """Calculates the average color in a patch around the center point."""
    x, y = int(center[0]), int(center[1])
    h, w, _ = image.shape
    x1 = max(0, x - size)
    x2 = min(w, x + size)
    y1 = max(0, y - size)
    y2 = min(h, y + size)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.array([0, 0, 0])
    avg_color = patch.mean(axis=(0, 1))
    return avg_color

def save_tracks_to_json(trajectories, output_path: Path):
    """
    Saves player trajectories and their overall direction of movement to a JSON file.

    Args:
        trajectories (dict): A dictionary mapping player IDs to their track data.
        output_path (Path): The path to save the JSON file.
    """
    output_data = {}
    for pid, track in trajectories.items():
        if len(track) > 1:
            # Calculate overall direction vector from start to end point
            start_pos = np.array(track[0]['center'])
            end_pos = np.array(track[-1]['center'])
            direction_vector = (end_pos - start_pos).tolist()
        else:
            direction_vector = [0, 0]  # Not enough data to determine direction

        output_data[f"player_{pid}"] = {
            "overall_direction": direction_vector,
            "trajectory": track
        }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"[INFO] Saved player tracking data to {output_path}")




def annotate_video(input_path: Path, output_path: Path, model_path: Path):
    """
    Performs player detection and tracking on a video, saving an annotated video
    and a JSON file with tracking data.
    """
    model = YOLO(str(model_path))
    class_names = model.names

    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = get_fourcc(output_path)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # --- Tracking Configuration ---
    MAX_PLAYERS = 22
    MAX_DISAPPEARED_FRAMES = 30  # How many frames to wait for a player to reappear
    MAX_REID_DISTANCE = 150      # Max distance (pixels) to consider a match for a reappearing player
    VELOCITY_SMOOTHING = 0.6     # Smoothing factor for velocity calculation

    # --- Data Structures for Tracking ---
    player_data = {}             # Stores all data for a persistent player ID
    persistent_id_pool = set(range(MAX_PLAYERS)) # Available IDs (0-21)
    player_trajectories = defaultdict(list) # Stores the path for each player ID
    frame_number = 0

    with tqdm(total=total_frames, desc="Tracking Players", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_number += 1

            results = model.track(
                frame, persist=True, verbose=False, tracker='bytetrack.yaml'
            )[0]

            annotator = Annotator(frame, line_width=2, example=class_names)

            # --- Frame Detections ---
            current_detections = []
            for box in results.boxes:
                if int(box.cls) == 0:  # Class 0 is 'player'
                    current_detections.append({
                        'box': box.xyxy[0].tolist(),
                        'center': get_center(box.xyxy[0]), # This now returns JSON-safe floats
                        'color': get_avg_color(frame, get_center(box.xyxy[0])),
                        'tracker_id': int(box.id.item()) if box.id is not None else None,
                        'pid': None  # Persistent ID to be assigned
                    })

            # --- Matching Logic ---
            # 1. Match using active tracker_id from ByteTrack
            for det in current_detections:
                if det['tracker_id'] is not None:
                    for pid, data in player_data.items():
                        if data['tracker_id'] == det['tracker_id']:
                            det['pid'] = pid
                            break

            # 2. Re-identify occluded players using position, direction, and color
            unmatched_detections = [d for d in current_detections if d['pid'] is None]
            disappeared_pids = [pid for pid, data in player_data.items() if data['disappeared'] > 0]

            if unmatched_detections and disappeared_pids:
                # Greedily match unmatched detections to disappeared players
                for det in unmatched_detections:
                    best_score = float('inf')
                    best_pid = None
                    for pid in disappeared_pids:
                        if pid in [d['pid'] for d in current_detections if d['pid'] is not None]:
                            continue # Skip if this PID was already matched

                        player = player_data[pid]
                        # Predict next position based on last velocity
                        predicted_pos = np.array(player['pos']) + np.array(player['vel'])
                        dist = euclidean_distance(det['center'], predicted_pos)

                        if dist < MAX_REID_DISTANCE:
                            color_dist = np.linalg.norm(det['color'] - player['color'])
                            score = dist + 0.5 * color_dist  # Weighted score
                            if score < best_score:
                                best_score = score
                                best_pid = pid

                    if best_pid is not None:
                        det['pid'] = best_pid

            # --- Update Player States ---
            # 1. Update matched and re-identified players
            for det in current_detections:
                pid = det['pid']
                if pid is not None:
                    old_pos = player_data[pid]['pos']
                    new_pos = det['center']

                    # Calculate and smooth velocity
                    new_vel = (np.array(new_pos) - np.array(old_pos)).tolist()
                    smoothed_vel = ( (1 - VELOCITY_SMOOTHING) * np.array(player_data[pid]['vel']) + VELOCITY_SMOOTHING * np.array(new_vel) ).tolist()

                    player_data[pid].update({
                        'pos': new_pos,
                        'vel': smoothed_vel,
                        'color': det['color'],
                        'tracker_id': det['tracker_id'],
                        'disappeared': 0
                    })
                    player_trajectories[pid].append({'frame': frame_number, 'center': new_pos})

            # 2. Assign new IDs to new players
            for det in current_detections:
                if det['pid'] is None:
                    if persistent_id_pool:
                        new_pid = min(persistent_id_pool)
                        persistent_id_pool.remove(new_pid)
                        det['pid'] = new_pid

                        player_data[new_pid] = {
                            'pos': det['center'],
                            'vel': [0, 0], 'color': det['color'],
                            'disappeared': 0, 'tracker_id': det['tracker_id']
                        }
                        player_trajectories[new_pid].append({'frame': frame_number, 'center': det['center']})

            # 3. Handle players that have disappeared
            current_matched_pids = {d['pid'] for d in current_detections if d['pid'] is not None}
            for pid in list(player_data.keys()):
                if pid not in current_matched_pids:
                    player_data[pid]['disappeared'] += 1
                    player_data[pid]['tracker_id'] = None # Tracker ID is lost
                    if player_data[pid]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                        persistent_id_pool.add(pid) # Return ID to the pool
                        del player_data[pid]

            # --- Annotation ---
            # Draw players (class 0)
            for det in current_detections:
                if det['pid'] is not None:
                    label = f"Player ID:{det['pid']}"
                    annotator.box_label(det['box'], label, color=colors(0, bgr=True))

            # Draw referees (class 1)
            for box in results.boxes:
                if int(box.cls) == 1:
                    annotator.box_label(box.xyxy[0], f"{class_names[1]}", color=colors(1, bgr=True))

            out.write(annotator.result())
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save tracking data to JSON
    json_output_path = output_path.parent / f"{input_path.stem}_tracks.json"
    save_tracks_to_json(player_trajectories, json_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track American football players in a video.")
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument("--output", help="Path to save the annotated output video file.")
    parser.add_argument("--model", help="Path to the YOLOv8 model file (.pt).")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    # Default output path if not provided
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path("./output").resolve() / f"{input_path.stem}_tracked.avi"

    # Default model path if not provided
    if args.model:
        model_path = Path(args.model).resolve()
    else:
        model_path = Path("./models/american_v2.pt") # Make sure you have this model

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    annotate_video(input_path, output_path, model_path)