#omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam omsairam 

import os
import json
import pandas as pd

from .build import DATASET_REGISTRY
from .base_ds import BaseDataset

@DATASET_REGISTRY.register()
class EgoPer(BaseDataset):
    """
    EgoPer dataset loader for egocentric cooking segmentation,
    modeled after FineGym but using your custom annotation structure.
    """

    def __init__(self, cfg, mode):
        super(EgoPer, self).__init__(cfg, mode)
        self._construct_loader()

    def _construct_loader(self):
        annotation_path = self.cfg.DATA.ANNOTATION_FILE
        videos_dir = self.cfg.DATA.PATH_TO_DATA_DIR

        with open(annotation_path, "r") as f:
            all_recipe_data = json.load(f)
            data = all_recipe_data["coffee"] #change to dynamic 

        rows = []
        for recipe_data in data.items():
            for seg in recipe_data["segments"]:
                video_id = seg["video_id"]
                labels = seg["labels"]
                actions = labels["action"]
                action_types = labels.get("action_type", [None] * len(actions))
                time_stamps = labels["time_stamp"]
                error_desc = labels.get("error_description", [None] * len(actions))
                video_path = os.path.join(videos_dir, f"{video_id}.mp4")

                for i in range(len(actions)):
                    rows.append({
                        "dataset" : 'Egoper', 
                        "video_id": video_id,
                        "video_path": video_path,
                        "action": actions[i],
                        "action_type": action_types[i],
                        "start": time_stamps[i][0],
                        "end": time_stamps[i][1],
                        "description": error_desc[i],
                    })

        self.dataset_df = pd.DataFrame(rows)

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        seg = self.dataset_df.iloc[idx]
        # Direct access for Trokens: returns dict for this segment
        return seg.to_dict()


if __name__ == "__main__":
    # Direct test for egocentric cooking data as in FineGym
    class DummyCfg:
        class DATA:
            PATH_TO_DATA_DIR = "./sample_videos"
            ANNOTATION_FILE = "./sample_recipe_annotations.json"
    cfg = DummyCfg()
    mode = "train"

    # Only write sample file if it doesn't exist for test/demo
    if not os.path.exists(cfg.DATA.ANNOTATION_FILE):
        sample_ann = {
            "coffee": {
                "segments": [
                    {
                        "video_id": "coffee_test_001",
                        "labels": {
                            "action": [0, 1],
                            "action_type": [0, 1],
                            "time_stamp": [[0.0, 2.7], [2.8, 11.3]],
                            "error_description": ["BG", "Pour water"]
                        }
                    }
                ]
            }
        }
        os.makedirs(cfg.DATA.PATH_TO_DATA_DIR, exist_ok=True)
        with open(cfg.DATA.ANNOTATION_FILE, "w") as f:
            json.dump(sample_ann, f)

    ds = EgoPer(cfg, mode)
    print(f"Total segments loaded: {len(ds)}")
    for i in range(len(ds)):
        print(f"Segment {i}: {ds[i]}")