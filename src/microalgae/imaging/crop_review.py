"""Persistent proposals, human review, and traceable training exports."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
from shutil import copyfile
import stat
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4
import cv2
import numpy as np
import yaml
from .detection import detect_regions
from .dataset import IMAGE_EXTENSIONS
DEFAULT = dict(name='hysteresis', low_ratio=0.4, background_kernel_size=15, min_component_area=160, max_side=512, margin_ratio=0.1)
DECISIONS = ('pending', 'Accept target', 'Non-target alga', 'Unsure / mixed', 'Discard crop')
ARTIFACT = 'Artifact'
TARGET_CLASSES = ('m_aeruginosa', 'n_fruticosa', 'n_palea')

class HysteresisDetector:

    def __init__(self, settings):
        self.settings = {'max_side': DEFAULT['max_side'], 'margin_ratio': DEFAULT['margin_ratio']}
        self.settings.update((k, v) for k, v in settings.items() if k != 'name')
        for key in ('background_kernel_size', 'min_component_area', 'max_side'):
            if type(self.settings[key]) is not int or self.settings[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        kernel = self.settings['background_kernel_size']
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError('background_kernel_size must be odd and at least 3')
        for key, allow_zero in (('low_ratio', False), ('margin_ratio', True)):
            value = self.settings[key]
            if type(value) not in (float, int) or not 0 <= value <= 1 or (not allow_zero and value == 0):
                raise ValueError(f'{key} is outside its valid range')

    def detect(self, image):
        return detect_regions(image, **self.settings)['regions']
DETECTORS = {'hysteresis': HysteresisDetector}

def make_detector(settings):
    if not isinstance(settings, dict) or not isinstance(settings.get('name'), str):
        raise ValueError('Detector settings need a detector name')
    name = settings['name']
    if name not in DETECTORS:
        raise ValueError(f'Detector {name!r} is unavailable in this app')
    return DETECTORS[name](settings)


def load_detector_settings(path):
    settings = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if isinstance(settings, dict) and settings.get('name') == DEFAULT['name']:
        unknown = settings.keys() - DEFAULT.keys()
        if unknown:
            raise ValueError(f'Unknown detector settings: {", ".join(sorted(map(str, unknown)))}')
        settings = DEFAULT | settings
    make_detector(settings)
    return settings

def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f'Cannot decode {path}')
    return image

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read_review(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Unsupported crop review version')
    return data

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
        self.data = _read_review(self.path)
        self.saved_hash = digest(self.path)
        if self.data.get('pending_crop_deletions'):
            self.save()

    @classmethod
    def _create(cls, directory, data):
        directory.mkdir(parents=True, exist_ok=False)
        batch = cls.__new__(cls)
        batch.directory = directory
        batch.path = directory / 'review.json'
        batch.data = data
        batch.saved_hash = None
        return batch

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
        data = dict(schema_version=1, root=str(root), settings=settings, collection=collection, group_evidence='Dataset owner confirms each input is a distinct original and specimen', images=[])
        batch = cls._create(directory, data)
        for path in paths:
            path = Path(path).resolve()
            identifier = path.relative_to(root).as_posix()
            row = dict(image_id=identifier, parent_image_id=identifier, target=path.parent.name, sha256=digest(path), pilot_partition=partitions.get(identifier), reviewed=False, status='pending', error=None, crops=[])
            data['images'].append(row)
            try:
                for region, crop in batch._propose(row, path, detector):
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
            except Exception as error:
                row['status'], row['error'] = ('failed', str(error))
        batch.save()
        return batch

    def import_class_folder(self, folder):
        from PIL import Image

        folder = Path(folder)
        target = next((label for label in TARGET_CLASSES if label == folder.name.casefold()), 'unknown')
        class_name = target if target != 'unknown' else folder.name
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        paths = sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in extensions),
                       key=lambda path: (path.name.casefold(), path.name))
        rows = self.data['images']
        known_ids = {row['image_id'].casefold() for row in rows}
        root_name = Path(self.data['root']).name
        for row in rows:
            identifier = Path(row['image_id'])
            if len(identifier.parts) == 1:
                known_ids.add(f'{root_name}/{identifier.name}'.casefold())
            elif row.get('batch_local_source') and 'source_path' not in row and row['target'] != 'unknown':
                known_ids.add(f"{row['target']}/{identifier.name}".casefold())
        detector = make_detector(self.data['settings'])
        added = []
        duplicates = 0
        failures = []
        with TemporaryDirectory(prefix='.import-', dir=self.directory) as temporary:
            staging = Path(temporary)
            for path in paths:
                identifier = f'{class_name}/{path.name}'
                identity = identifier.casefold()
                if identity in known_ids:
                    duplicates += 1
                    continue
                try:
                    key = uuid4().hex
                    draft = staging / key
                    relative = Path('sources') / key / path.name
                    source = draft / relative
                    source.parent.mkdir(parents=True)
                    copyfile(path, source)
                    sha256 = digest(source)
                    with Image.open(source, formats=['PNG', 'JPEG', 'TIFF']) as reader:
                        if reader.format == 'TIFF' and reader.n_frames > 1:
                            raise ValueError('Multi-page TIFF is not supported')
                    row = dict(image_id=identifier, parent_image_id=identifier, target=target,
                               sha256=sha256, batch_local_source=True, source_path=relative.as_posix(),
                               pilot_partition=None,
                               reviewed=False, status='pending', error=None, crops=[])
                    for _ in self._propose(row, source, detector, output_directory=draft):
                        pass
                except Exception as error:
                    failures.append(f'{path.name}: {error}')
                    continue
                known_ids.add(identity)
                added.append((draft, row))

            if added:
                self._commit_import(rows, added)
        return duplicates, failures

    def _commit_import(self, rows, added):
        previous_hash = self.saved_hash
        first_new = len(rows)
        published = []
        source_folders = []
        try:
            for draft, row in added:
                source = self.directory / row['source_path']
                source.parent.mkdir(parents=True)
                source_folders.append(source.parent)
                for relative in (row['source_path'], *(crop['path'] for crop in row['crops'])):
                    destination = self.directory / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    (draft / relative).rename(destination)
                    published.append(destination)
            rows.extend(row for _, row in added)
            self.save()
        except Exception:
            if self.saved_hash == previous_hash:
                del rows[first_new:]
                for path in published:
                    path.unlink(missing_ok=True)
                for folder in source_folders:
                    folder.rmdir()
            raise

    def _propose(self, row, path, detector, *, output_directory=None):
        image = read_image(path)
        row['size'] = [image.shape[1], image.shape[0]]
        for region in detector.detect(image):
            crop = self.add(row, region['box'], region['crop_box'], automatic=True,
                            image=image, output_directory=output_directory)
            yield region, crop
        row['status'] = 'proposals' if row['crops'] else 'zero proposals'

    def source(self, row):
        root = self.directory if row.get('batch_local_source') else Path(self.data['root'])
        path = root / row.get('source_path', row['image_id'])
        if digest(path) != row['sha256']:
            raise ValueError('Source image changed')
        return read_image(path)

    def add(self, row, box, crop_box=None, replaces=None, automatic=False, image=None, *, output_directory=None):
        image = self.source(row) if image is None else image
        box = list(map(int, box))
        if crop_box is None:
            margin = max(2, int(np.ceil(max(box[2] - box[0], box[3] - box[1]) * self.data['settings'].get('margin_ratio', 0.1))))
            crop_box = [max(0, box[0] - margin), max(0, box[1] - margin), min(image.shape[1], box[2] + margin), min(image.shape[0], box[3] + margin)]
        identifier = uuid4().hex
        crop = dict(crop_id=identifier, box=box, crop_box=crop_box, automatic=automatic, replaces=replaces, active=True, decision='pending', path=f'crops/{identifier}.png')
        directory = self.directory if output_directory is None else output_directory
        write_crop(directory / crop['path'], image, crop_box)
        if replaces:
            next((c for c in row['crops'] if c['crop_id'] == replaces))['active'] = False
        row['crops'].append(crop)
        row['reviewed'] = False
        return crop

    def decide(self, row, crop, decision):
        if decision not in DECISIONS and not (decision == ARTIFACT and self.data['collection'] == 'evaluation'):
            raise ValueError('Unknown decision')
        if decision == 'Discard crop':
            self.discard(row, crop)
            return
        crop['decision'] = decision
        if decision != 'Accept target':
            crop.pop('classification', None)
        row['reviewed'] = False

    def classify(self, crop, classifier):
        if not crop['active'] or crop['decision'] != 'Accept target':
            raise ValueError('Review the crop and accept it before predicting a species')
        result = self.predict_crop(crop, classifier)
        previous = crop.get('classification')
        saved_hash = self.saved_hash
        crop['classification'] = result
        try:
            self.save()
        except Exception:
            if self.saved_hash == saved_hash:
                if previous is None:
                    del crop['classification']
                else:
                    crop['classification'] = previous
            raise

    def predict_crop(self, crop, classifier):
        result = {'model': classifier.name, 'fingerprint': classifier.fingerprint}
        try:
            result['species'] = classifier.predict(read_image(self.directory / crop['path']))
        except Exception as error:
            result['error'] = str(error)
        return result

    def discard(self, row, crop):
        self.remove_crops({crop['crop_id']}, reset_review=True)

    def deletion_path(self, relative):
        path = (self.directory / relative).resolve()
        crop_root = (self.directory / 'crops').resolve()
        if not path.is_relative_to(crop_root) or path == crop_root:
            raise ValueError('Deletion path outside crop directory')
        return path

    def _retained_deletion_paths(self, crops):
        crop_root = (self.directory / 'crops').resolve()
        linked_names = set()
        try:
            with os.scandir(crop_root) as entries:
                for entry in entries:
                    if entry.is_symlink() or (
                        os.name == 'nt'
                        and entry.stat(follow_symlinks=False).st_file_attributes
                        & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    ):
                        linked_names.add(os.path.normcase(entry.name))
        except FileNotFoundError:
            if crop_root.exists():
                raise

        for crop in crops:
            relative = crop['path']
            # Generated names address direct children and cannot be Windows short names.
            match = re.fullmatch(r'crops/([0-9a-f]{32}\.png)', relative)
            if match and os.path.normcase(match[1]) not in linked_names:
                yield crop_root / match[1]
            else:
                yield self.deletion_path(relative)

    def remove_crops(self, identifiers, reset_review=False):
        from copy import deepcopy
        if digest(self.path) != self.saved_hash:
            raise ValueError('Batch changed in another window; reopen it')
        removed = [c for r in self.data['images'] for c in r['crops'] if c['crop_id'] in identifiers]
        paths = {self.deletion_path(c['path']) for c in removed}
        retained = [c for r in self.data['images'] for c in r['crops'] if c['crop_id'] not in identifiers]
        if paths.intersection(self._retained_deletion_paths(retained)):
            raise ValueError('Discarded crop file is shared by another crop')
        before = deepcopy(self.data)
        for row in self.data['images']:
            affected = any(c['crop_id'] in identifiers for c in row['crops'])
            row['crops'] = [c for c in row['crops'] if c['crop_id'] not in identifiers]
            if affected and reset_review:
                row['reviewed'] = False
            for crop in row['crops']:
                if crop.get('replaces') in identifiers:
                    crop['replaces'] = None
        pending = set(self.data.get('pending_crop_deletions', []))
        directory = self.directory.resolve()
        pending.update(path.relative_to(directory).as_posix() for path in paths)
        if pending:
            self.data['pending_crop_deletions'] = sorted(pending)
        previous_hash = self.saved_hash
        try:
            self.save()
        except Exception:
            if self.saved_hash == previous_hash:
                self.data = before
            raise
        return len(removed)

    def finish_deletions(self):
        pending = self.data.get('pending_crop_deletions', [])
        if not pending:
            return
        paths = {self.deletion_path(relative) for relative in pending}
        retained = (c for row in self.data['images'] for c in row['crops'])
        if paths.intersection(self._retained_deletion_paths(retained)):
            raise ValueError('Pending deletion references a retained crop')
        for path in paths:
            path.unlink(missing_ok=True)
        del self.data['pending_crop_deletions']
        self._write_review()

    def complete(self, row):
        if row['status'] == 'failed' or any((c['active'] and c['decision'] == 'pending' for c in row['crops'])):
            raise ValueError('Resolve every active proposal before completing this image')
        row['reviewed'] = True

    def save(self):
        self._write_review()
        self.finish_deletions()

    def _write_review(self):
        payload = json.dumps(self.data, indent=2).encode('utf-8')
        with NamedTemporaryFile(dir=self.directory, prefix='.review-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            temporary.write_bytes(payload)
            if self.saved_hash is not None and digest(self.path) != self.saved_hash:
                raise ValueError('Batch changed in another window; reopen it')
            temporary.replace(self.path)
            self.saved_hash = hashlib.sha256(payload).hexdigest()
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def audit(directory):
        directory = Path(directory)
        # Opening a CropBatch can recover pending deletions. An audit only reads metadata.
        data = _read_review(directory / 'review.json')
        images = data.get('images')
        if not isinstance(images, list) or any(
            not isinstance(row, dict) or not isinstance(row.get('crops'), list)
            or any(not isinstance(crop, dict) for crop in row['crops'])
            for row in images
        ):
            raise ValueError('Invalid image or crop records in review.json')

        with TemporaryDirectory(prefix='.audit-', dir=directory) as staging:
            source_csv = Path(staging) / 'source_inventory.csv'
            crop_csv = Path(staging) / 'crop_manifest.csv'
            with source_csv.open('w', encoding='utf-8', newline='') as stream:
                fields = ['image_id', 'parent_image_id', 'target', 'sha256', 'pilot_partition', 'status', 'error', 'reviewed']
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: row.get(key) for key in fields} for row in images)

            with crop_csv.open('w', encoding='utf-8', newline='') as stream:
                fields = ['image_id', 'parent_image_id', 'crop_id', 'box', 'crop_box', 'path', 'decision', 'active', 'replaces']
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in images:
                    for crop in row['crops']:
                        writer.writerow({key: row.get(key, crop.get(key)) for key in fields})

            source_csv.replace(directory / source_csv.name)
            crop_csv.replace(directory / crop_csv.name)

    def export(self, directory):
        if self.data['collection'] != 'training':
            raise ValueError('Evaluation collections cannot enter training exports')
        directory = Path(directory).resolve()
        if directory.is_relative_to(Path(self.data['root']).resolve()) or directory.is_relative_to(self.directory.resolve()):
            raise ValueError('Export must be outside source and review directories')
        selected = [(r, c) for r in self.data['images'] if r['reviewed'] for c in r['crops'] if c['active'] and c['decision'] == 'Accept target']
        if not selected:
            raise ValueError('No reviewed target crops')
        classes = TARGET_CLASSES
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
