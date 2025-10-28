#omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam  
"""
Point tracking module for extracting and tracking semantic points in videos.

This module provides functionality to:
- Extract semantic points from videos using clustering methods
- Track points across video frames using CoTracker
- Save tracking results and generate visualizations
"""
import sys
import os
import time
import random
import argparse
import pickle
import torch
import numpy as np
from einops import rearrange
import pandas as pd

# Clear GPU memory at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


from utils import convert_points_for_tracking, save_video
from feat_extractor import feature_extract
from get_semantic_points import get_points_from_clustering
from new_video_loader import load_video_pyvideo_reader
from omni_vis import vis_trail
import gc 

def log_memory_usage(stage, log_file="memory_log.txt", start_time=None, end_time=None):
    """Log detailed memory usage to file"""
    import psutil
    import datetime
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
    else:
        gpu_allocated = gpu_reserved = gpu_max = 0
    
    # CPU memory
    cpu_percent = psutil.virtual_memory().percent
    cpu_used = psutil.virtual_memory().used / 1e9
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add time segment info if provided
    time_segment = ""
    if start_time is not None and end_time is not None:
        time_segment = f" | Segment: {start_time}s-{end_time}s"
    elif start_time is not None:
        time_segment = f" | Start: {start_time}s"
    elif end_time is not None:
        time_segment = f" | End: {end_time}s"
    
    log_msg = f"{timestamp} | {stage:<25}{time_segment} | GPU: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved, {gpu_max:.2f}GB peak | CPU: {cpu_used:.2f}GB ({cpu_percent:.1f}%)\n"
    
    # Print to console
    print(log_msg.strip())
    
    # Write to file
    with open(log_file, "a") as f:
        f.write(log_msg)

def log_failed_video(index, video_path, error_msg, base_path, duration=None, num_frames=None, start_time=None, end_time=None):
    """Log failed video information to a dedicated error log file"""
    import datetime
    import cv2
    
    # Get video info if not provided
    if duration is None or num_frames is None:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_calc = total_frames / fps if fps > 0 else 0
                cap.release()
                if duration is None:
                    duration = duration_calc
                if num_frames is None:
                    num_frames = total_frames
            else:
                duration = duration or "Unknown"
                num_frames = num_frames or "Unknown"
        except:
            duration = duration or "Unknown"
            num_frames = num_frames or "Unknown"
    
    # GPU memory info
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_max = torch.cuda.max_memory_allocated() / 1e9
    else:
        gpu_allocated = gpu_reserved = gpu_max = 0
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_name = video_path.split('/')[-1]
    
    error_log_path = os.path.join(base_path, "failed_videos_log.txt")
    
    # Format start and end time for logging
    time_segment = ""
    if start_time is not None and end_time is not None:
        time_segment = f" | Segment: {start_time}s-{end_time}s"
    elif start_time is not None:
        time_segment = f" | Start: {start_time}s"
    elif end_time is not None:
        time_segment = f" | End: {end_time}s"
    
    log_entry = (f"{timestamp} | Index: {index} | Video: {video_name}{time_segment} | "
                f"Duration: {duration}s | Frames: {num_frames} | "
                f"GPU: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved, {gpu_max:.2f}GB peak | "
                f"Error: {error_msg}\n")
    
    print(f"FAILED VIDEO LOGGED: {video_name}")
    
    # Write to error log
    with open(error_log_path, "a") as f:
        f.write(log_entry)

# set seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = '/fs/cfar-projects/actionloc/camera_ready/tats_v2/dumps'

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
# pylint: disable=redefined-outer-name


def check_columns_in_df(df):
    """Check if the dataframe has the required columns.

    Args:
        df (pd.DataFrame): Dataframe to check

    Raises:
        ValueError: If the dataframe does not have the required columns
    """
    required_columns = ['video_path', 'dataset']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the dataframe")


def extract_points(args, cotracker, feat_extractor, video_path, ds_dump_path, start_time, end_time, 
                    custom_fps, index):
    """Extract points from a video and save them to a pickle file.

    Args:
        args (argparse.Namespace): Arguments
        cotracker (torch.nn.Module): Cotracker model
        feat_extractor (torch.nn.Module): Feature extractor model
        video_path (str): Path to the video
        ds_dump_path (str): Path to the directory where the pickle file will be saved
        custom_fps (int): Custom fps to use for the video if video duration > 90s

    Returns:
        bool: True if the points were extracted, False otherwise
    """
    try:
        log_memory_usage(f"Video {index} - Start", start_time=start_time, end_time=end_time)
        
        # load video for DINO feat extractor
        vid_name = video_path.split('/')[-1].split('.')[0]
        debug_vis_dump_root = os.path.join(ds_dump_path, 'debug_vis', vid_name)
        feat_dump_path = os.path.join(ds_dump_path, 'feat_dump', f'{vid_name + "_S" + str(start_time) + "_E" + str(end_time)}.pkl')
        gif_dump_path = os.path.join(ds_dump_path, 'gif_dump', f'{vid_name + "_S" + str(start_time) + "_E" + str(end_time)}.gif')

        print("Feature dump path: ", feat_dump_path)
        if os.path.exists(feat_dump_path) and not args.rerun:
            log_memory_usage(f"Video {index} - Skipped (exists)")
            return True

        log_memory_usage(f"Video {index} - Loading video")
        video_loaded, video_frames, frames_id_dict = load_video_pyvideo_reader(
            video_path, return_tensor=True, use_float=False,
            num_frames=args.num_frames_clustering, sample_all_frames=False,
            fps=custom_fps if custom_fps is not None else args.fps, seg_start_time=start_time, seg_end_time=end_time)  # (B, T, C, H, W)
        if not video_loaded:
            print(f"Video {vid_name} not loaded")
            return None
        video_frames = rearrange(video_frames, 'b t c h w -> b t h w c')
        video_frames = video_frames.cpu().numpy()

        log_memory_usage(f"Video {index} - Video loaded")

        if args.debug_mode:
            time_start = time.time()

        log_memory_usage(f"Video {index} - Starting clustering")
        base_point_info = get_points_from_clustering(
            args, video_frames, feat_extractor, debug_vis_dump_root)
        
        points_list, point_labels_list, component_labels_list = base_point_info
        log_memory_usage(f"Video {index} - Clustering done")

        queries_points, cluster_ids_all_frames = convert_points_for_tracking(
            points_list, point_labels_list, frames_id_dict=frames_id_dict,
            component_labels_list=component_labels_list,
            use_connected_components=args.use_connected_components, device=args.device)
        
        log_memory_usage(f"Video {index} - Points converted")

        if args.debug_mode:
            os.makedirs(debug_vis_dump_root, exist_ok=True)
            time_end = time.time()
            print(f"Time taken to get points and labels: {time_end - time_start} seconds")
        torch.cuda.empty_cache()

        log_memory_usage(f"Video {index} - Loading full video")
        _, video, _ = load_video_pyvideo_reader(video_path, return_tensor=True, use_float=True,
                                 device=args.device, sample_all_frames=True,
                                 fps=custom_fps if custom_fps is not None else args.fps, seg_start_time=start_time, seg_end_time=end_time)  # B T C H W
        log_memory_usage(f"Video {index} - Full video loaded")
        
        if args.debug_mode:
            time_start = time.time()
            
        log_memory_usage(f"Video {index} - Starting CoTracker")
        
        # Aggressive memory cleanup before CoTracker to avoid fragmentation
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()
        
        # Use no_grad context to prevent gradient accumulation
        with torch.no_grad():
            if args.use_grid:
                pred_tracks, pred_visibility = cotracker(
                    video, grid_size=args.cotracker_grid_size,
                    queries=None, backward_tracking=False)
            else:
                pred_tracks, pred_visibility = cotracker(video,
                                                         queries=queries_points,
                                                         backward_tracking=True)
        log_memory_usage(f"Video {index} - CoTracker done")
        
        if args.debug_mode:
            time_end = time.time()
            print(f"Time taken to run cotracker: {time_end - time_start} seconds")
        point_queries = queries_points.cpu().squeeze(0).numpy()[:, 0]
        pred_tracks = pred_tracks.cpu().squeeze(0).numpy()
        pred_visibility = pred_visibility.cpu().squeeze(0).numpy()
        video = video.cpu().squeeze(0).numpy()
        video = rearrange(video, 't c h w -> t h w c')
        pt_obj_cluster_dict = {}

        dump_dict = {
            'pred_tracks': torch.tensor(pred_tracks).half(),
            'pred_visibility': torch.tensor(pred_visibility).bool(),
            'obj_ids': torch.tensor(cluster_ids_all_frames).long(),
            'point_queries': torch.tensor(point_queries).long(),
            **pt_obj_cluster_dict
        }

        os.makedirs(os.path.dirname(feat_dump_path), exist_ok=True)
        pickle.dump(dump_dict, open(feat_dump_path, "wb"))
        torch.cuda.empty_cache()
        
        log_memory_usage(f"Video {index} - Saved pickle")

        if args.debug_mode or args.make_vis:
            frames = vis_trail(video, pred_tracks, pred_visibility,
                               cluster_ids=cluster_ids_all_frames)
            os.makedirs(os.path.dirname(gif_dump_path), exist_ok=True)
            save_video(frames, gif_dump_path)
            
        log_memory_usage(f"Video {index} - Complete", start_time=start_time, end_time=end_time)
        return True
        
    except Exception as e:
        # Log failed video with detailed info
        log_failed_video(index, video_path, str(e), args.base_feat_path, None, None, start_time, end_time)
        log_memory_usage(f"Video {index} - ERROR: {str(e)}", start_time=start_time, end_time=end_time)
        print(f"Error processing video {vid_name}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_mode", action="store_true",
                        help="Enable debug mode")

    parser.add_argument("--use_connected_components", action="store_true",
                        help="Use connected components")

    parser.add_argument("--num_frames_clustering", type=int, default=32,
                        help="Number of frames to cluster")

    parser.add_argument("--merge_ratio", type=int, default=25,
                        help="Merge ratio")

    parser.add_argument("--num_iters", type=int, default=11,
                        help="Number of iterations")

    parser.add_argument("--clustering_method", type=str, default='bipartite',
                        help="Clustering method to use")

    parser.add_argument("--n_clusters", type=int, default=32,
                        help="Number of clusters")

    parser.add_argument("--num_points_per_entity", type=int, default=16,
                        help="Number of samples per mask")

    parser.add_argument("--use_grid", action="store_true",
                        help="Use grid")

    parser.add_argument("--cotracker_grid_size", type=int, default=16,
                        help="Cotracker grid size")

    parser.add_argument("--csv_path", type=str, default='sample.csv',
                        help='Path to csv file')

    parser.add_argument("--fps", type=int, default=None,
                        help="FPS for point tracking")

    parser.add_argument("--base_feat_path", type=str, default=BASE_PATH,
                        help="Base path for feature dumps")

    parser.add_argument("--make_vis", action="store_true",
                        help="Make gifs")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun the point tracking")

    args = parser.parse_args()

    use_connected_components = args.use_connected_components

    if args.clustering_method == 'kmeans':
        CLUSTER_STR = f'kmeans_n{args.n_clusters}'
    elif args.clustering_method == 'bipartite':
        CLUSTER_STR = 'bip'
    else:
        raise ValueError(f"Invalid clustering method: {args.clustering_method}")
    df = pd.read_csv(args.csv_path)
    df = df.sort_values(by='duration', ascending=True)
    #df = df.head(100)
    check_columns_in_df(df)
    if args.debug_mode:
        df = df.iloc[:1]  # just running on the first sample for debugging

    dump_name = f'cotracker3_{CLUSTER_STR}_fr_{args.num_frames_clustering}'
    if args.merge_ratio != 25 or args.num_iters != 11:  # if not default then add to dump name
        dump_name += f'_m{args.merge_ratio}_i{args.num_iters}'
    if use_connected_components:
        dump_name += '_concomp'
    #if args.fps is not None:
        #dump_name += f'_fps_{args.fps}'

    
    # Clear GPU memory at startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


    # base_featpath = '/fs/cfar-projects/actionloc/shirley/sam_based_debug/somethingv2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(args, 'device', device)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    feat_extractor = feature_extract()
    index = 0   
    new_video_uniq_id = None 

    total_count = 0 
    for video_index, vid_info_row in df.iterrows():
        dataset = vid_info_row['dataset']
        video_path = vid_info_row['video_path']
        start_time = vid_info_row.get('start_time', None)
        end_time = vid_info_row.get('end_time', None)
        if 'duration' in vid_info_row:
            duration = vid_info_row['duration']
            #if duration>90:
             #   custom_fps = 1
            #if duration:
            #   print(duration)
            #    continue 
            #else:
            #    custom_fps = None
        else:
            custom_fps = None
        #print(custom_fps)

         
        video_uniq_id = video_path.split('/')[-1].split('.')[0]
        #if new_video_uniq_id is not None and new_video_uniq_id != video_uniq_id:
        #    index = 0
        new_video_uniq_id = video_uniq_id
        feat_dump_name = f'{video_uniq_id}'
        ds_dump_path = os.path.join(args.base_feat_path, dump_name, dataset)
       # print("Extracting points for video:", video_path, index)
        index += 1
        total_count += 1 
        
        # Log video processing start with timing info
        time_info = ""
        if start_time is not None and end_time is not None:
            time_info = f" (Segment: {start_time}s-{end_time}s)"
        elif start_time is not None:
            time_info = f" (Start: {start_time}s)"
        elif end_time is not None:
            time_info = f" (End: {end_time}s)"
        
        print(index)

        #Coffee/oatmeal: ran first 780 videos with fps=10
        #1000 for pinwheels 


        #NOTE: RAN UNTIL 4250, now running until 4400 
         


        print("made it here")


        custom_fps = 10

        print(f"Processing video {index}: {video_path.split('/')[-1]}{time_info} with custom fps={custom_fps}")

        extract_points(args, cotracker, feat_extractor, video_path, ds_dump_path, start_time, end_time, 
                        custom_fps, index)
        
        # Force cleanup between videos
        torch.cuda.empty_cache()
        gc.collect() 
        print("MEMORY ISSUES:")
        print(torch.cuda.memory_summary())




#ERROR AT 8783 
            

