"""Persistent proposals, human review, and traceable training exports."""
import csv
import hashlib
import json
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
import yaml
from .detection import detect_regions
from .dataset import IMAGE_EXTENSIONS
DEFAULT = dict(name='hysteresis', low_ratio=0.4, background_kernel_size=15, min_component_area=160, max_side=512, margin_ratio=0.1)
DECISIONS = ('pending', 'Accept target', 'Non-target alga', 'Unsure / mixed', 'Reject artifact')

class HysteresisDetector:

    def __init__(self, settings):
        self.settings = {k: v for k, v in settings.items() if k != 'name'}

    def detect(self, image):
        return detect_regions(image, **self.settings)['regions']
DETECTORS = {'hysteresis': HysteresisDetector}

def make_detector(settings):
    return DETECTORS[settings['name']](settings)

def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return image

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_crop(path, image, box):
    l, t, r, b = box
    if not (0 <= l < r <= image.shape[1] and 0 <= t < b <= image.shape[0]):
        raise ValueError('Crop outside source image')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image[t:b, l:r])
    if not ok:
        raise OSError('PNG encoding failed')
    encoded.tofile(path)

class CropBatch:

    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / 'review.json'
        self.data = json.loads(self.path.read_text())
        self.saved_hash = digest(self.path)
        if self.data['schema_version'] != 1:
            raise ValueError('Unsupported crop review version')

    @classmethod
    def prepare(cls, root, directory, settings=None, files=None, pilot=None, collection='training'):
        root, directory = (Path(root).resolve(), Path(directory).resolve())
        if directory == root or directory.is_relative_to(root):
            raise ValueError('Review directory must be outside source dataset')
        settings = dict(DEFAULT if settings is None else settings)
        detector = make_detector(settings)
        partitions = {}
        references = {}
        if pilot:
            references = {r['image_id']: r for r in json.loads(Path(pilot).read_text())['images']}
            partitions = {key: r['partition'] for key, r in references.items()}
        paths = sorted(files if files is not None else (p for p in root.glob('*/*') if p.suffix.lower() in IMAGE_EXTENSIONS))
        directory.mkdir(parents=True, exist_ok=False)
        data = dict(schema_version=1, root=str(root), settings=settings, collection=collection, group_evidence='Dataset owner confirms each input is a distinct original and specimen', images=[])
        batch = object.__new__(cls)
        batch.directory, batch.path, batch.data, batch.saved_hash = (directory, directory / 'review.json', data, None)
        for path in paths:
            path = Path(path).resolve()
            identifier = path.relative_to(root).as_posix()
            row = dict(image_id=identifier, parent_image_id=identifier, target=path.parent.name, sha256=digest(path), pilot_partition=partitions.get(identifier), reviewed=False, status='pending', error=None, crops=[])
            data['images'].append(row)
            try:
                image = read_image(path)
                row['size'] = [image.shape[1], image.shape[0]]
                for region in detector.detect(image):
                    crop = batch.add(row, region['box'], region['crop_box'], automatic=True, image=image)
                    matches = []
                    for obj in references.get(identifier, {}).get('objects', []):
                        a, b = (region['box'], obj['box'])
                        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
                        if union and intersection / union >= 0.5 and (not obj['uncertain']):
                            matches.append(obj)
                    if len(matches) == 1:
                        species = matches[0]['species']
                        crop['suggestion'] = 'Accept target' if species == row['target'] else 'Non-target alga' if species == 'other_alga' else 'Unsure / mixed'
                row['status'] = 'proposals' if row['crops'] else 'zero proposals'
            except Exception as error:
                row['status'], row['error'] = ('failed', str(error))
        batch.save()
        return batch

    def source(self, row):
        path = Path(self.data['root']) / row['image_id']
        if digest(path) != row['sha256']:
            raise ValueError('Source image changed')
        return read_image(path)

    def add(self, row, box, crop_box=None, replaces=None, automatic=False, image=None):
        image = self.source(row) if image is None else image
        box = list(map(int, box))
        if crop_box is None:
            margin = max(2, int(np.ceil(max(box[2] - box[0], box[3] - box[1]) * self.data['settings'].get('margin_ratio', 0.1))))
            crop_box = [max(0, box[0] - margin), max(0, box[1] - margin), min(image.shape[1], box[2] + margin), min(image.shape[0], box[3] + margin)]
        identifier = uuid4().hex
        crop = dict(crop_id=identifier, box=box, crop_box=crop_box, automatic=automatic, replaces=replaces, active=True, decision='pending', path=f'crops/{identifier}.png')
        write_crop(self.directory / crop['path'], image, crop_box)
        if replaces:
            next((c for c in row['crops'] if c['crop_id'] == replaces))['active'] = False
        row['crops'].append(crop)
        row['reviewed'] = False
        return crop

    def decide(self, row, crop, decision):
        if decision not in DECISIONS:
            raise ValueError('Unknown decision')
        crop['decision'] = decision
        row['reviewed'] = False

    def complete(self, row):
        if row['status'] == 'failed' or any((c['active'] and c['decision'] == 'pending' for c in row['crops'])):
            raise ValueError('Resolve every active proposal before completing this image')
        row['reviewed'] = True

    def save(self):
        if self.saved_hash is not None and digest(self.path) != self.saved_hash:
            raise ValueError('Batch changed in another window; reopen it')
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, indent=2))
        temporary.replace(self.path)
        self.saved_hash = digest(self.path)
        with (self.directory / 'source_inventory.csv').open('w', newline='') as stream:
            fields = ['image_id', 'parent_image_id', 'target', 'sha256', 'pilot_partition', 'status', 'error', 'reviewed']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(({key: row.get(key) for key in fields} for row in self.data['images']))
        fields = ['image_id', 'parent_image_id', 'crop_id', 'box', 'crop_box', 'path', 'decision', 'active', 'replaces']
        with (self.directory / 'crop_manifest.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.data['images']:
                for crop in row['crops']:
                    writer.writerow({key: row.get(key, crop.get(key)) for key in fields})

    def export(self, directory):
        if self.data['collection'] != 'training':
            raise ValueError('Evaluation collections cannot enter training exports')
        directory = Path(directory).resolve()
        if directory.is_relative_to(Path(self.data['root']).resolve()) or directory.is_relative_to(self.directory.resolve()):
            raise ValueError('Export must be outside source and review directories')
        selected = [(r, c) for r in self.data['images'] if r['reviewed'] for c in r['crops'] if c['active'] and c['decision'] == 'Accept target']
        if not selected:
            raise ValueError('No reviewed target crops')
        classes = ['m_aeruginosa', 'n_fruticosa', 'n_palea']
        if any((r['target'] not in classes for r, c in selected)):
            raise ValueError('Choose a supported target species before export')
        directory.mkdir(parents=True, exist_ok=False)
        for label in classes:
            (directory / 'images' / label).mkdir(parents=True)
        mapping = []
        for row, crop in selected:
            relative = f"{row['target']}/{crop['crop_id']}.png"
            write_crop(directory / 'images' / relative, self.source(row), crop['crop_box'])
            mapping.append([relative, row['parent_image_id']])
        with (directory / 'parent_groups.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'parent_image_id'])
            writer.writerows(mapping)
        (directory / 'review_snapshot.json').write_text(json.dumps(self.data, indent=2))
        root = Path(__file__).resolve().parents[3]
        config = yaml.safe_load((root / 'configs/training.yaml').read_text())
        config['dataset']['raw_dir'] = str(directory / 'images')
        config['dataset']['parent_groups_csv'] = str(directory / 'parent_groups.csv')
        (directory / 'training.yaml').write_text(yaml.safe_dump(config))
        return len(mapping)
