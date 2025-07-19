import cv2
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

def get_fourcc(output_path: Path):
    ext = output_path.suffix.lower()
    if ext == '.mp4':
        return cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        return cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError(f"Unsupported video format: {ext}")

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return [float((x1 + x2) / 2), float((y1 + y2) / 2)]

def get_avg_color(image, center, size=10):
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
    output_data = {}
    for pid, track in trajectories.items():
        if len(track) > 1:
            start_pos = np.array(track[0]['center'])
            end_pos = np.array(track[-1]['center'])
            direction_vector = (end_pos - start_pos).tolist()
        else:
            direction_vector = [0, 0]
        output_data[f"player_{pid}"] = {
            "overall_direction": direction_vector,
            "trajectory": track
        }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"[INFO] Saved player tracking data to {output_path}")

def get_largest_connected_mask(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255

def annotate_video(input_path: Path, output_path: Path, model_path: Path):
    model = YOLO(str(model_path))
    seg_model = YOLO(str(Path("./models/field_seg.pt")))
    class_names = model.names

    cap = cv2.VideoCapture(str(input_path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = get_fourcc(output_path)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Tracking parameters
    MAX_PLAYERS = 22
    MAX_DISAPPEARED_FRAMES = 15  # Increased from 10
    MAX_REID_DISTANCE = 150  # Reduced from 450
    VELOCITY_SMOOTHING = 0.4
    VELOCITY_WEIGHT = 60.0
    APPEARANCE_WEIGHT = 0.4  # New weight for appearance similarity
    MOTION_WEIGHT = 0.6     # New weight for motion similarity
    MIN_CONFIDENCE = 0.7    # Minimum confidence for re-identification

    player_data = {}
    persistent_id_pool = set(range(MAX_PLAYERS))
    player_trajectories = defaultdict(list)
    player_appearances = defaultdict(list)  # Store appearance features
    frame_number = 0

    with tqdm(total=total_frames, desc="Tracking Players", unit="frame") as pbar:
        while cap.isOpened():
            ret, original_frame = cap.read()
            if not ret:
                break
            frame_number += 1
            frame = original_frame.copy()

            # Field segmentation
            seg_results = seg_model(original_frame, verbose=False)[0]
            masks = seg_results.masks
            classes = seg_results.boxes.cls if seg_results.boxes is not None else []

            field_mask = np.zeros((height, width), dtype=np.uint8)
            if masks is not None and masks.data.shape[0] > 0:
                for i, mask in enumerate(masks.data.cpu().numpy()):
                    if int(classes[i]) != 0:
                        continue
                    resized = cv2.resize(mask, (width, height))
                    binary = (resized > 0.5).astype(np.uint8) * 255
                    field_mask = cv2.bitwise_or(field_mask, binary)

            main_field_mask = get_largest_connected_mask(field_mask)

            row_sums = np.sum(main_field_mask, axis=1)
            row_sums = np.convolve(row_sums, np.ones(5) / 5, mode='same')

            valid_rows = np.where(row_sums > np.mean(row_sums))[0]
            if len(valid_rows) >= 2:
                top_y = valid_rows[0]
                bottom_y = valid_rows[-1]
                cv2.rectangle(frame, (0, top_y), (width - 1, bottom_y), (0, 255, 255), 2)
            else:
                top_y = 0
                bottom_y = height - 1
            # Player detection and tracking
            results = model.track(original_frame, persist=True, verbose=False, 
                                 tracker='botsort.yaml', conf=0.6)[0]
            annotator = Annotator(frame, line_width=2, example=class_names)

            current_detections = []
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls) == 0:  # Player class
                        center = get_center(box.xyxy[0])
                        x, y = int(center[0]), int(center[1])
                        
                        # Skip detections outside field or mask
                        if y < top_y or y > bottom_y or main_field_mask[y, x] == 0:
                            continue
                            
                        # Get appearance features (color histogram)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        player_patch = original_frame[y1:y2, x1:x2]
                        if player_patch.size > 0:
                            hist = cv2.calcHist([player_patch], [0, 1, 2], None, 
                                              [8, 8, 8], [0, 256, 0, 256, 0, 256])
                            hist = cv2.normalize(hist, hist).flatten()
                        else:
                            hist = np.zeros(8*8*8)
                            
                        current_detections.append({
                            'box': box.xyxy[0].tolist(),
                            'center': center,
                            'color': get_avg_color(original_frame, center),
                            'tracker_id': int(box.id.item()) if box.id is not None else None,
                            'pid': None,
                            'appearance': hist,
                            'confidence': box.conf.item()
                        })

            # First pass: Match current detections with existing players based on tracker_id
            for det in current_detections:
                if det['tracker_id'] is not None:
                    for pid, data in player_data.items():
                        if data['tracker_id'] == det['tracker_id']:
                            det['pid'] = pid
                            old_pos = player_data[pid]['pos']
                            new_pos = det['center']
                            new_vel = np.array(new_pos) - np.array(old_pos)
                            smoothed_vel = ((1 - VELOCITY_SMOOTHING) * np.array(player_data[pid]['vel']) +
                                            VELOCITY_SMOOTHING * new_vel).tolist()
                            
                            # Update appearance history (keep last 5 appearances)
                            player_appearances[pid].append(det['appearance'])
                            if len(player_appearances[pid]) > 5:
                                player_appearances[pid].pop(0)
                                
                            player_data[pid].update({
                                'pos': new_pos,
                                'vel': smoothed_vel,
                                'color': det['color'],
                                'tracker_id': det['tracker_id'],
                                'disappeared': 0,
                                'last_appearance': det['appearance']
                            })
                            player_trajectories[pid].append({'frame': frame_number, 'center': new_pos})
                            break
            
            # Second pass: Re-identify disappeared players using improved matching
            unmatched_dets = [d for d in current_detections if d['pid'] is None and d['confidence'] > MIN_CONFIDENCE]
            disappeared_pids = [pid for pid, data in player_data.items() if data['disappeared'] > 0 and 
                              data['disappeared'] <= MAX_DISAPPEARED_FRAMES]

            if unmatched_dets and disappeared_pids:
                cost_matrix = np.full((len(disappeared_pids), len(unmatched_dets)), float('inf'))
                
                for i, pid in enumerate(disappeared_pids):
                    player = player_data[pid]
                    
                    # Calculate predicted position
                    predicted_pos = np.array(player['pos']) + np.array(player['vel'])
                    
                    for j, det in enumerate(unmatched_dets):
                        # Motion similarity
                        dist_to_predicted = euclidean_distance(det['center'], predicted_pos)
                        
                        # Appearance similarity
                        if len(player_appearances[pid]) > 0:
                            avg_appearance = np.mean(player_appearances[pid], axis=0)
                            appearance_sim = cosine_similarity([avg_appearance], [det['appearance']])[0][0]
                            appearance_dist = 1 - appearance_sim
                        else:
                            appearance_dist = 0
                            
                        # Velocity direction similarity
                        det_velocity_candidate = np.array(det['center']) - np.array(player['pos'])
                        angle_diff = 0
                        if np.linalg.norm(player['vel']) > 0 and np.linalg.norm(det_velocity_candidate) > 0:
                            angle1 = np.arctan2(player['vel'][1], player['vel'][0])
                            angle2 = np.arctan2(det_velocity_candidate[1], det_velocity_candidate[0])
                            angle_diff = np.abs(angle1 - angle2)
                            if angle_diff > np.pi:
                                angle_diff = 2 * np.pi - angle_diff
                                
                        # Combined cost with weights
                        motion_cost = dist_to_predicted * (1 + angle_diff)
                        appearance_cost = appearance_dist * 100  # Scale to similar range as motion
                        
                        total_cost = (MOTION_WEIGHT * motion_cost + 
                                     APPEARANCE_WEIGHT * appearance_cost)
                        
                        if dist_to_predicted < MAX_REID_DISTANCE:
                            cost_matrix[i, j] = total_cost
                
                # Hungarian algorithm for optimal assignment
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    for i, j in zip(row_ind, col_ind):
                        if cost_matrix[i, j] < 150:  # Adjusted threshold
                            pid = disappeared_pids[i]
                            det = unmatched_dets[j]
                            det['pid'] = pid
                            
                            # Update velocity and appearance
                            new_vel = np.array(det['center']) - np.array(player_data[pid]['pos'])
                            smoothed_vel = ((1 - VELOCITY_SMOOTHING) * np.array(player_data[pid]['vel']) +
                                          VELOCITY_SMOOTHING * new_vel).tolist()
                                          
                            player_appearances[pid].append(det['appearance'])
                            if len(player_appearances[pid]) > 5:
                                player_appearances[pid].pop(0)
                            
                            player_data[pid].update({
                                'pos': det['center'],
                                'vel': smoothed_vel,
                                'color': det['color'],
                                'tracker_id': det['tracker_id'],
                                'disappeared': 0,
                                'last_appearance': det['appearance']
                            })
                            player_trajectories[pid].append({'frame': frame_number, 'center': det['center']})
                except ValueError:
                    pass

            # Assign new PIDs to truly new detections
            for det in current_detections:
                if det['pid'] is None and persistent_id_pool and det['confidence'] > MIN_CONFIDENCE:
                    new_pid = min(persistent_id_pool)
                    persistent_id_pool.remove(new_pid)
                    det['pid'] = new_pid
                    player_data[new_pid] = {
                        'pos': det['center'], 
                        'vel': [0, 0],
                        'color': det['color'], 
                        'disappeared': 0,
                        'tracker_id': det['tracker_id'],
                        'last_appearance': det['appearance']
                    }
                    player_appearances[new_pid] = [det['appearance']]
                    player_trajectories[new_pid].append({'frame': frame_number, 'center': det['center']})

            # Handle disappeared players
            matched_this_frame = {d['pid'] for d in current_detections if d['pid'] is not None}
            for pid in list(player_data.keys()):
                if pid not in matched_this_frame:
                    player_data[pid]['disappeared'] += 1
                    if player_data[pid]['disappeared'] > 2:
                        player_data[pid]['tracker_id'] = None  # Allow re-ID after 2 frames
                    if player_data[pid]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                        persistent_id_pool.add(pid)
                        del player_data[pid]
                        if pid in player_appearances:
                            del player_appearances[pid]

            # Annotate frame
            for det in current_detections:
                if det['pid'] is not None:
                    annotator.box_label(
                        det['box'],
                        f"Player {det['pid']}",
                        color=colors(det['pid'] % len(colors.palette), bgr=True)
                    )
                    start_point = (int(det['center'][0]), int(det['center'][1]))
                    player_velocity = player_data[det['pid']]['vel']
                    arrow_length_scale = 1.5 
                    end_point = (int(det['center'][0] + player_velocity[0] * arrow_length_scale),
                                 int(det['center'][1] + player_velocity[1] * arrow_length_scale))
                    
                    if np.linalg.norm(player_velocity) > 0.5:
                        cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 2, tipLength=0.5)

            # Draw ball if detected
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls) == 1:  # Ball class
                        annotator.box_label(box.xyxy[0], f"{class_names[1]}", color=colors(1, bgr=True))

            out.write(annotator.result())
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    json_output_path = output_path.parent / f"{input_path.stem}_tracks.json"
    save_tracks_to_json(player_trajectories, json_output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track American football players in a video.")
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument("--output", help="Path to save the annotated output video file.")
    parser.add_argument("--model", help="Path to the YOLO model file (.pt).")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path("./output").resolve() / f"{input_path.stem}_tracked.avi"

    if args.model:
        model_path = Path(args.model).resolve()
    else:
        model_path = Path("./models/american_v2.pt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    annotate_video(input_path, output_path, model_path)