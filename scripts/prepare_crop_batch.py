"""Prepare a persistent crop review batch or export reviewed targets."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import yaml
from microalgae.imaging.crop_review import CropBatch
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/raw')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default='configs/detection.yaml')
    parser.add_argument('--pilot', default='results/detection_rerun_20261115_07/split/annotations.json')
    parser.add_argument('--collection', choices=['training', 'evaluation'], default='training')
    parser.add_argument('--export-batch')
    args = parser.parse_args()
    if args.export_batch:
        print(CropBatch(args.export_batch).export(args.output_dir))
    else:
        batch = CropBatch.prepare(args.root, args.output_dir, yaml.safe_load(Path(args.config).read_text()), pilot=args.pilot, collection=args.collection)
        from collections import Counter
        print(Counter((r['status'] for r in batch.data['images'])))
