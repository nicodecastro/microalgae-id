"""Coordinate, approval and paired-score contracts for the isolated crop pilot."""

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from microalgae.imaging.preprocessing.configured import validate_preprocessing
from scripts import crop_geometry_pilot as pilot
from scripts._geometry_references import digest, rasterize
from scripts.diagnose_geometry import _reference_canvas, _save_png
from scripts.score_geometry_masks import overlap_scores


@pytest.mark.parametrize("shape", [(81, 137), (137, 81), (53, 75), (75, 53), (256, 256)])
def test_transform_matches_real_resize(shape):
    image = np.full((*shape, 3), 100, np.uint8)
    settings = validate_preprocessing({}, "geom")
    _, content = _reference_canvas(image, settings)
    np.testing.assert_array_equal(pilot.to_canvas(np.ones(shape, bool), pilot.grid_transform(shape)), content)


@pytest.mark.parametrize("shape,box", [((80, 120), [20, 10, 90, 70]), ((80, 120), [0, 0, 120, 80]),
                                     ((101, 73), [0, 0, 50, 90]), ((101, 73), [20, 40, 73, 101])])
def test_crop_mapping_support_and_fullframe_identity(shape, box):
    predicted = np.ones((256, 256), bool)
    actual = pilot.map_crop(predicted, box, shape)
    np.testing.assert_array_equal(actual, pilot.score_region(box, shape))
    if box == [0, 0, shape[1], shape[0]]:
        content = pilot.to_canvas(np.ones(shape, bool), pilot.grid_transform(shape))
        np.testing.assert_array_equal(actual, content)


@pytest.mark.parametrize("box", [[-1, 0, 8, 8], [0, 0, 11, 9], [2, 2, 2, 5], [0, 0, True, 5], [0., 0, 5, 5]])
def test_invalid_box(box):
    with pytest.raises(ValueError, match="Crop box"):
        pilot.validate_box(box, (10, 10))


def test_mapping_strips_crop_padding():
    shape, box = (100, 200), [40, 20, 160, 80]
    transform = pilot.grid_transform((60, 120))
    content = pilot.to_canvas(np.ones((60, 120), bool), transform)
    assert not pilot.map_crop(~content, box, shape).any()


@pytest.mark.parametrize('shape', [(53, 75), (75, 53), (81, 137), (137, 81)])
def test_full_frame_preserves_patterned_prediction_exactly(shape):
    y, x = np.indices((256, 256))
    pattern = ((x + 2 * y) % 7) < 3
    content = pilot.to_canvas(np.ones(shape, bool), pilot.grid_transform(shape))
    mapped = pilot.map_crop(pattern, [0, 0, shape[1], shape[0]], shape)
    np.testing.assert_array_equal(mapped, pattern & content)


def test_non_square_crop_maps_known_pixels_to_source_coordinates():
    # A 128x64 crop occupies a 256x128 strip centered vertically on its canvas.
    predicted = np.zeros((256, 256), bool)
    predicted[96:160, 64:192] = True
    actual = pilot.map_crop(predicted, [32, 64, 160, 128], (256, 256))
    expected = np.zeros_like(actual)
    expected[80:112, 64:128] = True
    np.testing.assert_array_equal(actual, expected)


def test_boundary_identical_empty_shift_and_independent_euclidean_matches():
    ref = np.zeros((30, 30), bool); ref[8:19, 9:20] = True
    domain = np.ones_like(ref)
    assert pilot.boundary_metrics(pilot.boundary_counts(ref, ref, domain))["boundary_fscore_1px"] == 1
    assert pilot.boundary_metrics(pilot.boundary_counts(np.zeros_like(ref), ref, domain))["boundary_fscore_2px"] == 0
    pred = np.roll(ref, 2, axis=1); pred[23:26, 23:26] = True
    result = pilot.boundary_counts(pred, ref, domain)
    interior = pilot.binary_erosion(domain, structure=pilot.CROSS, border_value=0)
    points = [np.argwhere(mask & ~pilot.binary_erosion(mask, structure=pilot.CROSS) & interior) for mask in (pred, ref)]
    distances = np.sqrt(((points[0][:, None] - points[1][None, :]) ** 2).sum(axis=2))
    for t in (1, 2, 3):
        assert result[f"matched_predicted_{t}px"] == (distances.min(axis=1) <= t).sum()
        assert result[f"matched_reference_{t}px"] == (distances.min(axis=0) <= t).sum()
    scores = pilot.boundary_metrics(result)
    assert scores['boundary_fscore_1px'] <= scores['boundary_fscore_2px'] <= scores['boundary_fscore_3px']


def test_artificial_domain_edges_do_not_earn_boundary_matches():
    ref = np.zeros((40, 40), bool); ref[5:35, 5:35] = True
    region = np.zeros_like(ref); region[15:25, 10:30] = True
    # Only reference's real vertical contours are outside this domain; clipping
    # the mask before finding contours would invent a matching rectangle.
    with pytest.raises(ValueError, match="no evaluable boundary"):
        pilot.boundary_counts(ref, ref, region)


def test_same_box_scores_and_empty_prediction():
    ref = np.zeros((30, 30), bool); ref[8:18, 8:18] = True
    region = np.zeros_like(ref); region[3:23, 3:23] = True
    pred = ref.copy(); pred[26:29, 26:29] = True
    assert overlap_scores(pred, ref, region) == overlap_scores(ref, ref, region)
    assert overlap_scores(np.zeros_like(ref), ref, region)['iou'] == 0


def test_missing_binary_mask_is_rejected_before_native_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, 'imread', lambda *args: pytest.fail('Missing path reached native decoder'))
    with pytest.raises(ValueError, match='Missing, nonbinary, or mismatched mask'):
        pilot.read_binary(tmp_path / 'missing.png', (256, 256))


@pytest.fixture
def synthetic_inputs(tmp_path, monkeypatch):
    raw_root = tmp_path / 'raw'; comparison = tmp_path / 'comparison'; references = tmp_path / 'references'
    references.mkdir(); comparison.mkdir()
    profiles = {}
    for profile, ratio in zip(pilot.PROFILES, (.5, .65)):
        profiles[profile] = validate_preprocessing({'foreground': {'enabled': True, 'method': 'otsu_background_hysteresis',
                                      'hysteresis': {'low_ratio': ratio}}}, 'geom')
    settings = {'dataset_dir': str(raw_root), 'profiles': profiles, 'baseline_preprocessing': validate_preprocessing({}, 'geom')}
    inputs = {'directory': comparison, 'reference_dir': references, 'settings': settings,
              'annotations': {}, 'inventory': {}, 'labels': {}, 'hashes': {}}
    for identifier, indices, _ in pilot.SELECTION:
        raw = np.full((80, 120, 3), 210, np.uint8)
        cv2.circle(raw, (60, 40), 15, (60, 60, 60), -1)
        path = raw_root / identifier; _save_png(path, raw)
        _, content = _reference_canvas(raw, settings['baseline_preprocessing'])
        vertices = [[96, 100], [160, 100], [160, 156], [96, 156]]
        polygons = [{'mode': 'add', 'vertices': vertices} for _ in range(max(indices) + 1)]
        source = {'image_id': identifier, 'sha256': digest(path), 'canvas_shape': [256, 256], 'polygons': polygons, 'status': 'approved'}
        reference = rasterize(polygons, content.shape)
        _save_png(references / 'masks' / (identifier + '_mask.png'), reference.astype(np.uint8) * 255)
        for profile in pilot.PROFILES:
            _save_png(comparison / 'previews' / profile / (identifier + '_image.png'), reference.astype(np.uint8) * 255)
        inputs['annotations'][identifier] = source
        inputs['inventory'][identifier] = {k: source[k] for k in ('image_id', 'sha256', 'canvas_shape')}
        inputs['labels'][identifier] = identifier.split('/')[0]
    inputs['hashes'] = {str(p.resolve()): digest(p) for directory in (raw_root, comparison, references)
                       for p in directory.rglob('*') if p.is_file()}
    monkeypatch.setattr(pilot, 'load_inputs', lambda: inputs)
    # Keep integration tests focused on data flow rather than figure rendering.
    monkeypatch.setattr(pilot, 'save_box_preview', lambda *args: None)
    monkeypatch.setattr(pilot, 'save_gallery', lambda *args: None)
    return inputs


def test_proposals_do_not_run_segmentation_and_pending_blocks_output(tmp_path, synthetic_inputs, monkeypatch):
    monkeypatch.setattr(pilot, 'prepare_feature_input', lambda *args: pytest.fail('Segmentation before approval'))
    folder = tmp_path / 'pilot'
    manifest = pilot.propose(folder)
    assert len(manifest['images']) == 9 and all(r['review_status'] == 'pending' for r in manifest['images'])
    assert not list(folder.rglob('*_mask.png'))
    with pytest.raises(ValueError, match='Unresolved'):
        pilot.run(folder, folder / 'evaluation')
    assert not (folder / 'evaluation').exists()


def approve(folder, manifest):
    for row in manifest['images']:
        row.update(review_status='approved', review_notes='Synthetic test approval')
    pilot.write_json(folder / 'crop_review.json', manifest)


def test_end_to_end_paired_coverage_summaries_and_preservation(tmp_path, synthetic_inputs):
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder); approve(folder, manifest)
    result = pilot.run(folder, folder / 'evaluation')
    assert len(result) == 36
    assert not result.duplicated(['image_id', 'profile', 'input']).any()
    assert len(pd.read_csv(folder / 'evaluation/paired_crop_changes.csv')) == 18
    assert len(pd.read_csv(folder / 'evaluation/paired_profile_changes.csv')) == 18
    perclass = pd.read_csv(folder / 'evaluation/summary_by_class.csv')
    macro = pd.read_csv(folder / 'evaluation/equal_species_summary.csv')
    for row in macro.itertuples():
        group = perclass[(perclass.profile == row.profile) & (perclass.input == row.input)]
        assert row.iou == pytest.approx(group.iou.mean())
    assert all(result[f'boundary_fscore_1px'] <= result[f'boundary_fscore_2px'])
    pilot.check_hashes(synthetic_inputs['hashes'])
    assert json.loads((folder / 'evaluation/run_status.json').read_text())['status'] == 'completed'
    assert json.loads((folder / 'evaluation/approved_manifest.json').read_text()) == manifest


def test_approved_crop_cutting_selected_reference_is_rejected(tmp_path, synthetic_inputs):
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder); approve(folder, manifest)
    manifest['images'][0]['box'] = [55, 36, 63, 44]
    pilot.write_json(folder / 'crop_review.json', manifest)
    with pytest.raises(ValueError, match='cuts selected'):
        pilot.run(folder, folder / 'evaluation')
    assert not (folder / 'evaluation').exists()


def test_reference_selection_cannot_include_another_group():
    annotation = {'canvas_shape': [30, 30], 'polygons': [
        {'mode': 'add', 'vertices': [[3, 3], [10, 3], [10, 10], [3, 10]]},
        {'mode': 'add', 'vertices': [[18, 18], [25, 18], [25, 25], [18, 25]]},
        {'mode': 'subtract', 'vertices': [[5, 5], [7, 5], [7, 7], [5, 7]]}]}
    selected = pilot.selected_reference(annotation, [0])
    assert selected[3, 3] and not selected[6, 6] and not selected[20, 20]
    with pytest.raises(ValueError, match='additive'):
        pilot.selected_reference(annotation, [2])


def test_rejected_cases_preserved_without_scores(tmp_path, synthetic_inputs):
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder); approve(folder, manifest)
    manifest['images'][0]['review_status'] = 'no_useful_crop'
    pilot.write_json(folder / 'crop_review.json', manifest)
    result = pilot.run(folder, folder / 'evaluation')
    assert len(result) == 32
    coverage = pd.read_csv(folder / 'evaluation/coverage.csv')
    assert len(coverage) == 9 and (coverage.processing_status == 'excluded_by_review').sum() == 1


def test_processing_error_is_explicit_and_paired_denominators_match(tmp_path, synthetic_inputs, monkeypatch):
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder); approve(folder, manifest)
    actual = pilot.prepare_feature_input
    count = 0
    def fail_once(*args):
        nonlocal count
        count += 1
        if count == 1:
            raise ValueError('Synthetic processing failure')
        return actual(*args)
    monkeypatch.setattr(pilot, 'prepare_feature_input', fail_once)
    with pytest.raises(RuntimeError, match='processing errors'):
        pilot.run(folder, folder / 'evaluation')
    frame = pd.read_csv(folder / 'evaluation/per_image_scores.csv')
    assert len(frame) == 32
    assert len(pd.read_csv(folder / 'evaluation/errors.csv')) == 1
    assert all(group.image_id.nunique() == 8 for _, group in frame.groupby(['profile', 'input']))


def test_mask_save_error_also_excludes_entire_image(tmp_path, synthetic_inputs, monkeypatch):
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder); approve(folder, manifest)
    actual = pilot._save_png
    failed = False
    def fail_one_common_mask(path, image):
        nonlocal failed
        if 'common_grid_predictions' in str(path) and not failed:
            failed = True
            raise OSError('Synthetic mask save failure')
        return actual(path, image)
    monkeypatch.setattr(pilot, '_save_png', fail_one_common_mask)
    with pytest.raises(RuntimeError, match='processing errors'):
        pilot.run(folder, folder / 'evaluation')
    frame = pd.read_csv(folder / 'evaluation/per_image_scores.csv')
    assert len(frame) == 32
    assert all(group.image_id.nunique() == 8 for _, group in frame.groupby(['profile', 'input']))
    assert len(pd.read_csv(folder / 'evaluation/errors.csv')) == 1


def test_coverage_check_rejects_missing_and_duplicate_conditions():
    records = [{'image_id': 'a', 'review_status': 'approved'}]
    rows = [{'image_id': 'a', 'profile': p, 'input': preparation}
            for p in pilot.PROFILES for preparation in ('full_image', 'crop')]
    frame = pd.DataFrame(rows)
    pilot.check_score_coverage(frame, records, [])
    with pytest.raises(ValueError, match='paired score coverage'):
        pilot.check_score_coverage(frame.iloc[:-1], records, [])
    with pytest.raises(ValueError, match='paired score coverage'):
        pilot.check_score_coverage(pd.concat([frame, frame.iloc[:1]]), records, [])


def test_changed_input_and_existing_output_rejected(tmp_path, synthetic_inputs):
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder); approve(folder, manifest)
    with pytest.raises(FileExistsError):
        pilot.propose(folder)
    path = Path(next(iter(synthetic_inputs['hashes'])))
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Protected input changed|Source image hash changed'):
        pilot.run(folder, folder / 'evaluation')
    assert not (folder / 'evaluation').exists()


def test_all_rejected_still_produces_readable_coverage(tmp_path, synthetic_inputs):
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder)
    for row in manifest['images']:
        row.update(review_status='no_useful_crop', review_notes='No usable rectangle')
    pilot.write_json(folder / 'crop_review.json', manifest)
    result = pilot.run(folder, folder / 'evaluation')
    assert result.empty
    for name in ('per_image_scores.csv', 'summary_by_class.csv', 'equal_species_summary.csv',
                 'paired_crop_changes.csv', 'paired_profile_changes.csv'):
        assert pd.read_csv(folder / 'evaluation' / name).empty


def test_approval_rejects_other_annotated_group_in_box(tmp_path, synthetic_inputs):
    inputs = synthetic_inputs
    identifier = pilot.SELECTION[0][0]
    # Add a separate target group outside the selected polygon, inside a full-frame box.
    inputs['annotations'][identifier]['polygons'].append(
        {'mode': 'add', 'vertices': [[30, 70], [50, 70], [50, 90], [30, 90]]})
    path = inputs['reference_dir'] / 'masks' / (identifier + '_mask.png')
    mask = rasterize(inputs['annotations'][identifier]['polygons'], (256, 256))
    _save_png(path, mask.astype(np.uint8) * 255)
    inputs['hashes'][str(path.resolve())] = digest(path)
    folder = tmp_path / 'pilot'; manifest = pilot.propose(folder); approve(folder, manifest)
    manifest['images'][0]['box'] = [0, 0, 120, 80]
    pilot.write_json(folder / 'crop_review.json', manifest)
    with pytest.raises(ValueError, match='Another annotated target'):
        pilot.run(folder, folder / 'evaluation')
    assert not (folder / 'evaluation').exists()


def test_gallery_renders_all_four_conditions(tmp_path, monkeypatch):
    raw = np.full((80, 120, 3), 180, np.uint8)
    settings = validate_preprocessing({}, 'geom')
    canvas, content = _reference_canvas(raw, settings)
    selected = np.zeros((256, 256), bool); selected[100:150, 100:150] = True
    scores = {**overlap_scores(selected, selected, content),
              **pilot.boundary_metrics(pilot.boundary_counts(selected, selected, content))}
    views = {label: (selected, scores) for label in ('Control / full_image', 'Control / crop', 'H65 / full_image', 'H65 / crop')}
    titles = []
    original_save = pilot.Figure.savefig
    def capture(figure, *args, **kwargs):
        titles.extend(ax.get_title() for ax in figure.axes)
        return original_save(figure, *args, **kwargs)
    monkeypatch.setattr(pilot.Figure, 'savefig', capture)
    path = tmp_path / 'gallery.png'
    pilot.save_gallery(path, 'synthetic', raw, [0, 0, 120, 80], canvas, selected, content, views)
    assert len(titles) == 6 and all(any(title.startswith(label) for title in titles) for label in views)
    assert cv2.imread(str(path)).shape[:2] == (1200, 1800)
