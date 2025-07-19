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

#  Define the path to your specific field image ---
FIELD_TEMPLATE_PATH = Path("./models/field.jpg")

# --- Parameters for the Bird's-Eye View Overlay ---
BIRDSEYE_OVERLAY_WIDTH_RATIO = 0.25 # Adjusted for a smaller overlay
BIRDSEYE_OVERLAY_ALPHA = 0.8
PLAYER_DOT_RADIUS = 12 # Increased radius to fit text
BALL_DOT_RADIUS = 5 # Slightly increased for visibility
PLAYER_DOT_COLOR = (0, 0, 255) # BGR format for red color
PLAYER_TEXT_COLOR = (255, 255, 255) # BGR format for white color
REF_DOT_COLOR = (139, 0, 0) # Dark Blue BGR for referee
REF_TEXT_COLOR = (255, 255, 255) # White text for referee

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
    y2 = min(h, x + size)
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

    # Load field image and prepare for overlay ---
    if not FIELD_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Field image not found at: {FIELD_TEMPLATE_PATH}. "
                                "Please ensure 'field.jpg' is in your 'models' directory.")
    field_image_original = cv2.imread(str(FIELD_TEMPLATE_PATH))
    if field_image_original is None:
        raise ValueError(f"Could not load field image from: {FIELD_TEMPLATE_PATH}")

    # Calculate target size for the overlay
    overlay_width = int(width * BIRDSEYE_OVERLAY_WIDTH_RATIO)
    overlay_height = int(field_image_original.shape[0] * (overlay_width / field_image_original.shape[1]))
    field_image_resized = cv2.resize(field_image_original, (overlay_width, overlay_height))

    # Determine overlay position (bottom center)
    overlay_x = (width - overlay_width) // 2
    overlay_y = height - overlay_height
    
    # Store homography matrix
    M = None 

    # Tracking parameters
    MAX_PLAYERS = 22
    MAX_DISAPPEARED_FRAMES = 15
    MAX_REID_DISTANCE = 150
    VELOCITY_SMOOTHING = 0.4
    VELOCITY_WEIGHT = 60.0
    APPEARANCE_WEIGHT = 0.4
    MOTION_WEIGHT = 0.6
    MIN_CONFIDENCE = 0.7

    player_data = {}
    referee_data = {} # To store referee data separately
    persistent_id_pool = set(range(MAX_PLAYERS))
    player_trajectories = defaultdict(list)
    player_appearances = defaultdict(list)
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
                    if int(classes[i]) != 0: # Assuming class 0 is the field
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
            else:
                top_y = 0
                bottom_y = height - 1
            
            #  Calculate homography once on the first frame if needed ---
            if frame_number == 1 and M is None:
                # Find min/max x for top_y and bottom_y rows where mask is active
                active_cols_top = np.where(main_field_mask[top_y, :] > 0)[0]
                active_cols_bottom = np.where(main_field_mask[bottom_y, :] > 0)[0]

                if len(active_cols_top) > 10 and len(active_cols_bottom) > 10:
                    src_pts = np.float32([
                        [active_cols_top.min(), top_y],
                        [active_cols_top.max(), top_y],
                        [active_cols_bottom.max(), bottom_y],
                        [active_cols_bottom.min(), bottom_y]
                    ])


                    temp_h, temp_w, _ = field_image_resized.shape
                    dst_pts = np.float32([
                        [0, 0],           # Top-left of the resized field image
                        [temp_w - 1, 0],   # Top-right
                        [temp_w - 1, temp_h - 1], # Bottom-right
                        [0, temp_h - 1]     # Bottom-left
                    ])
                    
                    if src_pts.shape[0] == 4 and dst_pts.shape[0] == 4:
                        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is None:
                            print("[WARNING] Homography calculation failed. Bird's-eye view may be incorrect.")
                    else:
                        print("[WARNING] Not enough robust points found for homography calculation. Bird's-eye view may be incorrect.")
                        M = None
                else:
                    print("[WARNING] Field mask too small or incomplete for initial homography estimation. Bird's-eye view may be incorrect.")
                    M = None
            
            # Player detection and tracking
            results = model.track(original_frame, persist=True, verbose=False, 
                                    tracker='botsort.yaml', conf=0.6)[0]
            annotator = Annotator(frame, line_width=2, example=class_names)

            current_detections = []
            player_positions_birdseye = []
            ball_position_birdseye = None
            referee_position_birdseye = None # To store referee position

            if results.boxes is not None:
                for box in results.boxes:
                    center = get_center(box.xyxy[0])
                    x, y = int(center[0]), int(center[1])
                    
                    cls_id = int(box.cls)

                    # Skip detections outside field or mask (for players and referees only)
                    if (cls_id == 0 or cls_id == 2) and (y < top_y or y > bottom_y or main_field_mask[y, x] == 0): # Assuming class 2 is referee
                        continue
                    
                    # Get appearance features (color histogram)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    patch = original_frame[y1:y2, x1:x2]
                    if patch.size > 0:
                        hist = cv2.calcHist([patch], [0, 1, 2], None, 
                                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                    else:
                        hist = np.zeros(8*8*8)
                        
                    det_info = {
                        'box': box.xyxy[0].tolist(),
                        'center': center,
                        'color': get_avg_color(original_frame, center),
                        'tracker_id': int(box.id.item()) if box.id is not None else None,
                        'pid': None,
                        'appearance': hist,
                        'confidence': box.conf.item(),
                        'cls': cls_id # Store class for ball/player/referee distinction
                    }
                    current_detections.append(det_info)

                    if M is not None:
                        # Project only if the center is within the valid field rows for players/referees
                        if (det_info['cls'] == 0 and y >= top_y and y <= bottom_y) or \
                            det_info['cls'] == 1:
                            pts = np.float32([[center[0], center[1]]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            birdseye_pos = dst[0,0].tolist()
                            
                            # Ensure projected points are within the bounds of the resized field image
                            birdseye_pos[0] = max(0, min(birdseye_pos[0], field_image_resized.shape[1] - 1))
                            birdseye_pos[1] = max(0, min(birdseye_pos[1], field_image_resized.shape[0] - 1))

                            if det_info['cls'] == 0: # Player
                                player_positions_birdseye.append({
                                    'pos': birdseye_pos,
                                    'color': colors(det_info['tracker_id'] % len(colors.palette), bgr=True) if det_info['tracker_id'] is not None else (255,255,255),
                                    'original_id': det_info['tracker_id'] # Store original tracker ID to link to player_data
                                })
                            elif det_info['cls'] == 1: # Referee
                                referee_position_birdseye = birdseye_pos

                                if 'referee' not in referee_data:
                                    referee_data['referee'] = {'pos': center, 'vel': [0, 0]}
                                else:
                                    old_pos_ref = referee_data['referee']['pos']
                                    new_vel_ref = np.array(center) - np.array(old_pos_ref)
                                    referee_data['referee']['pos'] = center
                                    referee_data['referee']['vel'] = ((1 - VELOCITY_SMOOTHING) * np.array(referee_data['referee']['vel']) +
                                                                       VELOCITY_SMOOTHING * new_vel_ref).tolist()
                                
            # First pass: Match current detections with existing players based on tracker_id
            for det in current_detections:
                if det['tracker_id'] is not None and det['cls'] == 0: # Only for players
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
            unmatched_dets = [d for d in current_detections if d['pid'] is None and d['confidence'] > MIN_CONFIDENCE and d['cls'] == 0]
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
                        appearance_cost = appearance_dist * 100
                        
                        total_cost = (MOTION_WEIGHT * motion_cost + 
                                      APPEARANCE_WEIGHT * appearance_cost)
                        
                        if dist_to_predicted < MAX_REID_DISTANCE:
                            cost_matrix[i, j] = total_cost
                
                # Hungarian algorithm for optimal assignment
                try:
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    for i, j in zip(row_ind, col_ind):
                        if cost_matrix[i, j] < 150:
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
                if det['pid'] is None and persistent_id_pool and det['confidence'] > MIN_CONFIDENCE and det['cls'] == 0:
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
                        player_data[pid]['tracker_id'] = None
                    if player_data[pid]['disappeared'] > MAX_DISAPPEARED_FRAMES:
                        persistent_id_pool.add(pid)
                        del player_data[pid]
                        if pid in player_appearances:
                            del player_appearances[pid]

            # Annotate main frame
            for det in current_detections:
                if det['cls'] == 0: # Players
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

                elif det['cls'] == 1: # Ball
                    annotator.box_label(det['box'], f"{class_names[1]}", color=colors(1, bgr=True))
                
                elif det['cls'] == 2: # Referee (assuming class 2 is referee)
                    annotator.box_label(det['box'], "Ref", color=REF_DOT_COLOR) # Using REF_DOT_COLOR for bounding box

            # Draw Bird's-Eye View Overlay ---
            # Start with the resized field image as the canvas
            birdseye_canvas = field_image_resized.copy()
            
            # Draw players on the bird's-eye canvas
            for player_dot in player_positions_birdseye:
                b_x, b_y = int(player_dot['pos'][0]), int(player_dot['pos'][1])
                # Ensure points are within the overlay bounds
                b_x = max(0, min(b_x, overlay_width - 1))
                b_y = max(0, min(b_y, overlay_height - 1))
                
                # Draw red circle for player
                cv2.circle(birdseye_canvas, (b_x, b_y), PLAYER_DOT_RADIUS, PLAYER_DOT_COLOR, -1)
                
                # Get player ID using the stored original_id and linking to player_data
                player_id = None
                for pid, data in player_data.items():
                    # We can use tracker_id to identify players if it's available and matched
                    if data.get('tracker_id') == player_dot.get('original_id') and player_dot.get('original_id') is not None:
                        player_id = pid
                        break
  


                if player_id is not None:
                    # Put player ID text
                    text = str(player_id)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5 # Further increased font scale for better visibility
                    thickness = 2 # Increased thickness
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Adjust text position to be more centered in the dot
                    text_x = b_x - text_width // 2
                    text_y = b_y + text_height // 3 # Adjusted y-position
                    
                    # Draw the text with a small black outline for better visibility
                    cv2.putText(birdseye_canvas, text, (text_x, text_y), 
                                font, font_scale, (0, 0, 0), thickness + 2) # Black outline
                    cv2.putText(birdseye_canvas, text, (text_x, text_y), 
                                font, font_scale, PLAYER_TEXT_COLOR, thickness) # White text
            

            # Draw Referee on the bird's-eye canvas ---
            # if referee_position_birdseye is not None:
            #     b_x, b_y = int(referee_position_birdseye[0]), int(referee_position_birdseye[1])
            #     b_x = max(0, min(b_x, overlay_width - 1))
            #     b_y = max(0, min(b_y, overlay_height - 1))
                
            #     cv2.circle(birdseye_canvas, (b_x, b_y), PLAYER_DOT_RADIUS, REF_DOT_COLOR, -1) # Dark blue for referee
                
            #     ref_text = "R"
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 1.0
            #     thickness = 2
            #     (text_width, text_height), _ = cv2.getTextSize(ref_text, font, font_scale, thickness)
                
            #     text_x = b_x - text_width // 2
            #     text_y = b_y + text_height // 3
                
            #     cv2.putText(birdseye_canvas, ref_text, (text_x, text_y), 
            #                 font, font_scale, (0, 0, 0), thickness + 2) # Black outline
            #     cv2.putText(birdseye_canvas, ref_text, (text_x, text_y), 
            #                 font, font_scale, REF_TEXT_COLOR, thickness) # White text for referee
            
            # Overlay the bird's-eye view onto the main frame
            target_x_end = min(overlay_x + overlay_width, width)
            target_y_end = min(overlay_y + overlay_height, height)
            
            overlay_region_width = target_x_end - overlay_x
            overlay_region_height = target_y_end - overlay_y

            if overlay_region_width > 0 and overlay_region_height > 0:
                # Get the region of the main frame where the overlay will be placed
                roi = frame[overlay_y:target_y_end, overlay_x:target_x_end]
                
                # Resize birdseye_canvas if its dimensions don't exactly match the ROI (due to clipping)
                birdseye_canvas_to_paste = birdseye_canvas[0:overlay_region_height, 0:overlay_region_width]

                # Blend the overlay with the main frame
                # The weighted sum is applied directly, making the field image and dots transparent
                frame[overlay_y:target_y_end, overlay_x:target_x_end] = cv2.addWeighted(
                    birdseye_canvas_to_paste, BIRDSEYE_OVERLAY_ALPHA,
                    roi, (1 - BIRDSEYE_OVERLAY_ALPHA), 0)


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

    # Ensure the models directory exists for the field image as well
    FIELD_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Check if field.jpg exists
    if not FIELD_TEMPLATE_PATH.exists():
        print(f"\n[CRITICAL WARNING] Required 'field.jpg' not found at: {FIELD_TEMPLATE_PATH}")
        print("Please place your bird's-eye view football field image named 'field.jpg' in the 'models' directory.")
        print("Without it, the bird's-eye view overlay will fail or be incorrect.\n")
    
    annotate_video(input_path, output_path, model_path)