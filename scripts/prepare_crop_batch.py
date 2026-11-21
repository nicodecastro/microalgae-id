"""Prepare or audit a crop review batch, or export reviewed targets."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from microalgae.imaging.crop_review import CropBatch, load_detector_settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/raw')
    parser.add_argument('--output-dir', required=True, help='destination directory, or existing review batch for --audit')
    parser.add_argument('--config', default='configs/detection.yaml')
    parser.add_argument('--pilot', default='results/detection_rerun_20261115_07/split/annotations.json')
    parser.add_argument('--collection', choices=['training', 'evaluation'], default='training')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--export-batch')
    mode.add_argument('--audit', action='store_true', help='refresh both audit CSVs in the existing review batch')
    args = parser.parse_args()
    if args.audit:
        try:
            CropBatch.audit(args.output_dir)
        except (OSError, ValueError) as error:
            parser.exit(1, f'Could not audit {args.output_dir}: {error}\n')
        print(f'Wrote source_inventory.csv and crop_manifest.csv in {args.output_dir}')
    elif args.export_batch:
        print(CropBatch(args.export_batch).export(args.output_dir))
    else:
        batch = CropBatch.prepare(args.root, args.output_dir, load_detector_settings(args.config), pilot=args.pilot, collection=args.collection)
        from collections import Counter
        print(Counter((r['status'] for r in batch.data['images'])))
