"""Persistent prediction runs over an ordered list of imported images."""

from pathlib import Path
from shutil import copyfile
from time import perf_counter

from .crop_review import CropBatch, digest, make_detector


def run_predictions(directory, images, settings, classifier, model_path, stop, progress):
    detector = make_detector(settings)
    directory = Path(directory)
    rows = [
        dict(image_id=identifier, parent_image_id=identifier, title=path.name,
             target='unknown', batch_local_source=True, reviewed=False,
             status='pending', error=None, crops=[])
        for identifier, path in images
    ]
    batch = CropBatch._create(directory, dict(
        schema_version=1, root=str(directory), settings=settings, collection='prediction',
        images=rows, classifier={
            'model': classifier.name, 'fingerprint': classifier.fingerprint, 'path': model_path,
        },
    ))
    batch.save()
    for index, (row, (_, original)) in enumerate(zip(rows, images), 1):
        if stop.is_set():
            break
        progress(index, None)
        started = perf_counter()
        relative = Path('sources') / row['image_id'] / original.name
        source = directory / relative
        try:
            source.parent.mkdir(parents=True)
            copyfile(original, source)
            row.update(source_path=relative.as_posix(), sha256=digest(source))
            for _ in batch._propose(row, source, detector):
                pass
        except Exception as error:
            row.update(status='failed', error=str(error))
        else:
            for crop in row['crops']:
                if stop.is_set():
                    row['status'] = 'stopped'
                    break
                crop['classification'] = batch.predict_crop(crop, classifier)
                batch.save()
            else:
                row['prediction_seconds'] = perf_counter() - started
        batch.save()
        progress(index, row)
    return batch


def rerun_edits(batch, classifier):
    if batch.data['collection'] != 'prediction' or (
            classifier.fingerprint != batch.data['classifier']['fingerprint']):
        raise ValueError('Choose the original classifier for this prediction run.')
    for crop in edited_crops(batch):
        crop['classification'] = batch.predict_crop(crop, classifier)
        batch.save()


def edited_crops(batch):
    return (crop for row in batch.data['images'] for crop in row['crops']
            if crop['active'] and not crop['automatic'] and not crop.get('classification', {}).get('species'))
