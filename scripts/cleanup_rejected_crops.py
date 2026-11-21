"""Remove legacy rejected crops from a batch and its exported review snapshot."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from microalgae.imaging.crop_review import CropBatch, digest


def cleanup(batch_dir, snapshot_path):
    batch = CropBatch(batch_dir)
    snapshot_path = Path(snapshot_path)
    snapshot_hash = digest(snapshot_path)
    snapshot = json.loads(snapshot_path.read_text())
    identifiers = {c['crop_id'] for data in [batch.data, snapshot] for row in data['images'] for c in row['crops'] if c['decision'] == 'Reject artifact'}
    # Validate batch paths before committing either metadata file.
    discarded_paths = {str(batch.deletion_path(c['path'])) for r in batch.data['images'] for c in r['crops'] if c['crop_id'] in identifiers}
    retained_paths = {str(batch.deletion_path(c['path'])) for r in batch.data['images'] for c in r['crops'] if c['crop_id'] not in identifiers}
    if discarded_paths.intersection(retained_paths):
        raise ValueError('Rejected crop shares a retained file')
    images = list((snapshot_path.parent/'images').glob('*/*.png'))
    if identifiers.intersection(path.stem for path in images):
        raise ValueError('Rejected crop is present in exported training images')
    protected = {str(p): digest(p) for p in images + [snapshot_path.parent/'parent_groups.csv']}
    reviewed = [row['reviewed'] for row in batch.data['images']]
    removed_snapshot = 0
    for row in snapshot['images']:
        removed_snapshot += sum(c['crop_id'] in identifiers for c in row['crops'])
        row['crops'] = [c for c in row['crops'] if c['crop_id'] not in identifiers]
        for crop in row['crops']:
            if crop.get('replaces') in identifiers:
                crop['replaces'] = None
    if digest(batch.path) != batch.saved_hash or digest(snapshot_path) != snapshot_hash:
        raise ValueError('Review data changed during cleanup')
    if removed_snapshot:
        temporary = snapshot_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot, indent=2))
        temporary.replace(snapshot_path)
    removed_batch = batch.remove_crops(identifiers)
    assert reviewed == [row['reviewed'] for row in batch.data['images']]
    assert all(digest(Path(path)) == value for path,value in protected.items())
    result = dict(batch_entries_removed=removed_batch, snapshot_entries_removed=removed_snapshot,
                  exported_training_images_verified=len(images), reviewed_images=sum(reviewed))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--snapshot', required=True)
    args = parser.parse_args()
    print(json.dumps(cleanup(args.batch, args.snapshot), indent=2))
