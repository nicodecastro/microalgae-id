import csv
import json
from pathlib import Path
import runpy
import sys
import cv2
import numpy as np
import pytest
from microalgae.imaging.crop_review import CropBatch, DEFAULT, DETECTORS, read_image

class Substitute:
    def __init__(self,settings): pass
    def detect(self,image):
        return [dict(box=[1,1,5,5],crop_box=[0,0,7,7])]


def test_import_image_class_folder_in_crop_review(tmp_path, qtbot, monkeypatch):
    from PIL import Image
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from microalgae.ui.crop_review import CropReviewWindow

    class ImportDetector(Substitute):
        def detect(self, image):
            value = image[0, 0, 0]
            if value == 12:
                return []
            if value == 13:
                raise ValueError('Detector failed')
            return super().detect(image)

    def image(folder, name, value):
        path = tmp_path / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (15, 12), (value,) * 3).save(path)
        return path

    monkeypatch.setitem(DETECTORS, 'import_test', ImportDetector)
    existing = image('raw/n_fruticosa', 'original.png', 10)
    other_class = image('raw/n_palea', 'a-new.png', 14)
    batch = CropBatch.prepare(existing.parent.parent, tmp_path / 'batch', {'name': 'import_test'},
                              files=[existing, other_class], collection='evaluation')
    first = image('outside/n_fruticosa', 'a-new.png', 11)
    empty = image('outside/n_fruticosa', 'b-zero.png', 12)
    image('outside/n_fruticosa', 'ORIGINAL.PNG', 14)
    broken = first.with_name('broken.png')
    broken.write_bytes(b'not an image')
    image('outside/n_fruticosa', 'failure.png', 13)
    image('outside/n_fruticosa/subfolder', 'ignored.png', 15)
    tiff = first.with_name('stack.tiff')
    frame = Image.new('RGB', (15, 12))
    frame.save(tiff, save_all=True, append_images=[frame])

    selection = ['']
    notices = []
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *_: selection[0])
    monkeypatch.setattr(QMessageBox, 'warning', lambda _, title, message: notices.append((title, message)))

    class Models:
        def add_menu(self, window):
            window.menuBar().addMenu('Models')

    window = CropReviewWindow(batch, models=Models())
    qtbot.addWidget(window)
    window.show()
    button = window.import_folder_button
    assert button.text() == 'Import image class folder'
    bar = window.menuBar()
    assert button.geometry().right() < bar.actionGeometry(bar.actions()[0]).left()
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert window.images.currentRow() == 0
    assert window._task is None

    selection[0] = str(first.parent)
    button.setFocus()
    qtbot.keyClick(button, Qt.Key.Key_Space)
    qtbot.waitUntil(lambda: window.images.count() == 4 and bool(notices))
    assert window.images.currentRow() == 2
    assert [window.images.item(i).text() for i in range(4)] == [
        'n_fruticosa/original.png', 'n_palea/a-new.png', 'n_fruticosa/a-new.png', 'n_fruticosa/b-zero.png',
    ]
    assert window.review_counter.text() == '0/4 reviewed'
    assert 'Imported 2 · Skipped 1 existing name(s) · Failed 3' in notices[0][1]
    assert 'Multi-page TIFF' in notices[0][1]
    assert not window.preview.pixmap().isNull()

    update = image('update/n_fruticosa', 'a-new.png', 16)
    selection[0] = str(update.parent)
    window.images.setCurrentRow(3)
    assert window.crops.count() == 0
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.statusBar().currentMessage().startswith('Imported 0 · Skipped 1'))
    assert window.images.currentRow() == 3

    renamed = update.with_name('c-new.png')
    renamed.write_bytes(first.read_bytes())
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.images.count() == 5)
    assert window.images.currentRow() == 4
    assert window.images.currentItem().text() == 'n_fruticosa/c-new.png'

    for path in (first, empty):
        path.rename(path.with_suffix('.moved'))
    reopened = CropBatch(batch.directory)
    proposed, zero, added_later = reopened.data['images'][2:]
    assert proposed['parent_image_id'] != zero['parent_image_id']
    assert proposed['target'] == zero['target'] == added_later['target'] == 'n_fruticosa'
    assert proposed['crops'][0]['decision'] == 'pending'
    assert zero['status'] == 'zero proposals'
    assert reopened.source(proposed)[0, 0, 0] == 11
    assert reopened.source(zero)[0, 0, 0] == 12
    restored = CropReviewWindow(reopened)
    qtbot.addWidget(restored)
    restored.images.setCurrentRow(2)
    assert not restored.preview.pixmap().isNull()


def test_import_class_folder_rolls_back_on_stale_save(tmp_path, monkeypatch):
    monkeypatch.setitem(DETECTORS, 'test', Substitute)
    root = tmp_path / 'n_fruticosa'
    root.mkdir()
    source = root / 'new.png'
    cv2.imwrite(str(source), np.full((12, 15, 3), 125, np.uint8))
    stale = CropBatch.prepare(root, tmp_path / 'batch', {'name': 'test'}, files=[])
    current = CropBatch(stale.directory)
    current.data['group_evidence'] = 'Updated by another window'
    current.save()

    with pytest.raises(ValueError, match='Batch changed in another window'):
        stale.import_class_folder(root)

    assert not stale.data['images']
    assert CropBatch(stale.directory).data == current.data
    assert sorted(path.name for path in stale.directory.rglob('*') if path.is_file()) == ['review.json']
    assert current.import_class_folder(root) == (0, [])
    assert len(CropBatch(stale.directory).data['images']) == 1


@pytest.fixture
def audit_cli(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_crop_batch.py"

    def run(directory, *extra_args):
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", [str(script), "--output-dir", str(directory), "--audit", *extra_args])
            patch.setattr(sys, "path", sys.path.copy())
            runpy.run_path(str(script), run_name="__main__")

    return run


def test_audit_refreshes_review_csvs(tmp_path, monkeypatch, audit_cli):
    monkeypatch.setitem(DETECTORS, "test", Substitute)
    root = tmp_path / "raw"
    root.mkdir()
    source = root / "image.png"
    assert cv2.imwrite(str(source), np.full((12, 15, 3), 120, np.uint8))
    directory = tmp_path / "batch"
    batch = CropBatch.prepare(root, directory, dict(name="test"), files=[source], collection="evaluation")
    assert {path.name for path in directory.iterdir()} == {"review.json", "crops"}

    row = batch.data["images"][0]
    original = row["crops"][0]
    replacement = batch.add(row, [2, 2, 6, 6], replaces=original["crop_id"])
    discarded = batch.add(row, [7, 2, 11, 6])
    batch.discard(row, discarded)
    assert CropBatch(directory).data == batch.data
    assert not (directory / discarded["path"]).exists()
    assert {path.name for path in directory.iterdir()} == {"review.json", "crops"}

    audit_cli(directory)
    inventory = directory / "source_inventory.csv"
    manifest = directory / "crop_manifest.csv"
    with inventory.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "image_id", "parent_image_id", "target", "sha256", "pilot_partition", "status", "error", "reviewed",
        ]
        assert [(entry["image_id"], entry["reviewed"]) for entry in reader] == [("image.png", "False")]
    with manifest.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "image_id", "parent_image_id", "crop_id", "box", "crop_box", "path", "decision", "active", "replaces",
        ]
        entries = list(reader)
        assert [entry["crop_id"] for entry in entries] == [original["crop_id"], replacement["crop_id"]]
        assert entries[0]["active"] == "False"
        assert entries[1]["replaces"] == original["crop_id"]
        assert entries[1]["box"] == "[2, 2, 6, 6]"

    previous_reports = inventory.read_bytes(), manifest.read_bytes()
    preparation_report = directory / "preparation_report.json"
    preparation_report.write_text("{}", encoding="utf-8")
    batch.decide(row, replacement, "Accept target")
    batch.complete(row)
    batch.save()
    assert (inventory.read_bytes(), manifest.read_bytes()) == previous_reports

    audit_cli(directory)
    assert b"True" in inventory.read_bytes()
    assert b"Accept target" in manifest.read_bytes()
    assert preparation_report.read_bytes() == b"{}"


@pytest.mark.parametrize("review", [None, "", "{", '{"schema_version": 1, "images": [{}]}'])
def test_audit_rejects_invalid_batch(tmp_path, audit_cli, review, capsys):
    directory = tmp_path / "batch"
    if review is not None:
        directory.mkdir()
    if review:
        (directory / "review.json").write_text(review, encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        audit_cli(directory)
    assert error.value.code == 1
    assert "Could not audit" in capsys.readouterr().err
    if directory.exists():
        assert {path.name for path in directory.iterdir()} == ({"review.json"} if review else set())
    assert directory.exists() == (review is not None)


def test_audit_and_export_batch_are_mutually_exclusive(tmp_path, audit_cli, capsys):
    with pytest.raises(SystemExit) as error:
        audit_cli(tmp_path, "--export-batch", "batch")
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_audit_stages_both_reports_before_replacing(tmp_path, monkeypatch):
    (tmp_path / "review.json").write_text('{"schema_version": 1, "images": []}', encoding="utf-8")
    reports = [tmp_path / name for name in ("source_inventory.csv", "crop_manifest.csv")]
    for report in reports:
        report.write_bytes(b"previous report")
    open_path = Path.open

    def fail_crop_report(path, mode="r", *args, **kwargs):
        if path.name == "crop_manifest.csv" and mode == "w":
            raise OSError("could not stage crop report")
        return open_path(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_crop_report)
    with pytest.raises(OSError, match="could not stage crop report"):
        CropBatch.audit(tmp_path)
    assert all(report.read_bytes() == b"previous report" for report in reports)
    assert {path.name for path in tmp_path.iterdir()} == {"review.json", *(report.name for report in reports)}


def test_review_export_and_substitution(tmp_path,monkeypatch):
    monkeypatch.setitem(DETECTORS,"test",Substitute)
    root=tmp_path/"raw"; (root/"n_palea").mkdir(parents=True)
    source=root/"n_palea/x.png"
    cv2.imwrite(str(source),np.full((12,15,3),125,np.uint8))
    batch=CropBatch.prepare(root,tmp_path/"batch",dict(name="test"))
    row=batch.data["images"][0]; original=row["crops"][0]
    crop=batch.add(row,[2,2,6,6],replaces=original["crop_id"])
    batch.add(row,[7,2,10,6])
    assert not original["active"]
    with pytest.raises(ValueError):batch.complete(row)
    for c in row["crops"]:
        if c["active"]:batch.decide(row,c,"Accept target")
    batch.complete(row);batch.save()
    reopened=CropBatch(tmp_path/"batch")
    assert reopened.data==batch.data
    assert batch.export(tmp_path/"export")==2
    assert (tmp_path/"export/parent_groups.csv").read_text().count("n_palea/x.png")==2
    for c in row["crops"]:
        l,t,r,b=c["crop_box"]
        assert np.array_equal(read_image(batch.directory/c["path"]),read_image(source)[t:b,l:r])
    stale=CropBatch(batch.directory);batch.save()
    batch.data["test"]=True;batch.save()
    with pytest.raises(ValueError):stale.save()


def test_pilot_parity():
    root=Path("results")
    if not (root/"crop_review_20261115_01/review.json").exists():pytest.skip("Local full batch not present")
    from microalgae.imaging.crop_review import make_detector
    references = json.loads((root/"detection_rerun_20261115_07/split/annotations.json").read_text())
    dataset = Path(references['dataset_dir'])
    detector = make_detector(DEFAULT)
    count = 0
    for stage, candidate in [("development", 5), ("evaluation", 0)]:
        directory = root/"detection_rerun_20261115_08"/stage
        for prediction in json.loads((directory/"predictions.json").read_text()):
            if prediction['candidate'] != candidate:
                continue
            image = read_image(dataset/prediction['image_id'])
            regions = detector.detect(image)
            assert regions == prediction['regions']
            for index, crop in enumerate(regions):
                name = Path(prediction['image_id'])
                path = directory/'crops'/name.parent/(name.name+f'__candidate_{candidate:02d}__crop_{index:03d}.png')
                l,t,r,b = crop['crop_box']
                assert np.array_equal(read_image(path), image[t:b,l:r])
            count += 1
    assert count == 30


def test_desktop_review(tmp_path,qtbot,monkeypatch):
    from microalgae.ui.crop_review import CropReviewWindow
    monkeypatch.setitem(DETECTORS,"test",Substitute)
    root=tmp_path/"raw";root.mkdir()
    cv2.imwrite(str(root/"x.jpg"),np.full((12,15,3),120,np.uint8))
    batch=CropBatch.prepare(root,tmp_path/"batch",dict(name="test"), files=list(root.glob("*")))
    window=CropReviewWindow(batch);qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(window.review_counter.isVisible)
    assert window.review_counter.geometry().height() >= 36
    assert window.review_counter.mapTo(window.centralWidget(), window.review_counter.rect().topLeft()).y() < window.images.mapTo(window.centralWidget(), window.images.rect().topLeft()).y()
    assert window.image_status.text() == "Current image · Proposals"
    assert window.review_counter.text() == "0/1 reviewed"
    assert window.crop["box"]==[1,1,5,5]
    assert window.statusBar().currentMessage() == "Original pixel orientation · proposals"
    crop_id = window.crop["crop_id"][:8]
    assert window.crops.currentItem().text() == f"pending · {crop_id}"
    window.decide("Non-target alga")
    assert window.crops.currentItem().text() == f"Non-target alga · {crop_id}"
    window.complete()
    assert window.statusBar().currentMessage() == "Original pixel orientation · Reviewed"
    assert window.image_status.text() == "Current image · Reviewed"
    assert window.review_counter.text() == "1/1 reviewed"
    assert CropBatch(batch.directory).data["images"][0]["reviewed"]
    reopened = CropReviewWindow(CropBatch(batch.directory));qtbot.addWidget(reopened)
    assert reopened.review_counter.text() == "1/1 reviewed"
    reopened.decide("Unsure / mixed")
    assert reopened.image_status.text() == "Current image · Proposals"
    assert reopened.review_counter.text() == "0/1 reviewed"


def test_rotated_jpeg_and_zero_proposals(tmp_path,qtbot,monkeypatch):
    from PIL import Image
    from microalgae.ui.crop_review import CropReviewWindow
    monkeypatch.setitem(DETECTORS,"test",Substitute)
    root=tmp_path/"raw";root.mkdir()
    exif=Image.Exif();exif[274]=6
    Image.fromarray(np.full((12,20,3),135,np.uint8)).save(root/"rotated.jpg",exif=exif)
    batch=CropBatch.prepare(root,tmp_path/"batch",dict(name="test"), files=list(root.glob("*")))
    window=CropReviewWindow(batch);qtbot.addWidget(window)
    assert window.viewer.sceneRect().width()==20
    assert window.viewer.sceneRect().height()==12
    assert window.preview.pixmap().isNull() is False
    row=batch.data["images"][0]
    batch.decide(row,row["crops"][0],"Non-target alga");batch.complete(row);batch.save()
    assert CropBatch(batch.directory).data["images"][0]["reviewed"]
    class Empty(Substitute):
        def detect(self,image):return []
    monkeypatch.setitem(DETECTORS,"empty",Empty)
    empty=CropBatch.prepare(root,tmp_path/"empty",dict(name="empty"),files=list(root.glob("*")))
    empty.complete(empty.data["images"][0]);empty.save()
    assert CropBatch(empty.directory).data["images"][0]["reviewed"]


def test_export_feature_audit_and_former_evaluation(tmp_path):
    from microalgae.imaging.dataset import build_dataset_manifest
    from training.extraction import extract_configured_features
    from training.feature_tables import audit_feature_tables,FEATURE_FILENAMES
    import yaml
    root=Path("results/crop_review_20261115_01")
    if not root.exists():pytest.skip("Local dataset not present")
    batch=CropBatch(root)
    row=next(r for r in batch.data["images"] if r["pilot_partition"]=="evaluation" and r["crops"])
    batch.data["images"]=[row]
    for crop in row["crops"]:batch.decide(row,crop,"Non-target alga")
    batch.decide(row,row["crops"][0],"Accept target");batch.complete(row)
    output=tmp_path/"export";batch.export(output)
    config=yaml.safe_load((output/"training.yaml").read_text())
    config["features"]["feature_list"]=["lbp"]
    extract_configured_features(config,tmp_path/"features",standardize=False)
    manifest=build_dataset_manifest(output/"images",config["dataset"]["classes"])
    assert audit_feature_tables(manifest,{"lbp":tmp_path/"features"/FEATURE_FILENAMES["lbp"]}).empty
    assert row["image_id"] in (output/"parent_groups.csv").read_text()


def test_page_keys_navigate_source_images(tmp_path, qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from microalgae.ui.crop_review import CropReviewWindow
    monkeypatch.setitem(DETECTORS, "test", Substitute)
    root = tmp_path / "raw"
    root.mkdir()
    for name in ["first.png", "second.png"]:
        cv2.imwrite(str(root / name), np.full((12, 15, 3), 120, np.uint8))
    batch = CropBatch.prepare(root, tmp_path / "batch", dict(name="test"), files=list(root.glob("*")))
    window = CropReviewWindow(batch)
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    window.crops.setFocus()
    qtbot.waitUntil(window.isActiveWindow)
    qtbot.keyClick(window.crops, Qt.Key.Key_PageDown)
    assert window.images.currentRow() == 1
    qtbot.keyClick(window.crops, Qt.Key.Key_PageDown)
    assert window.images.currentRow() == 1
    qtbot.keyClick(window.crops, Qt.Key.Key_PageUp)
    assert window.images.currentRow() == 0
    qtbot.keyClick(window.crops, Qt.Key.Key_PageUp)
    assert window.images.currentRow() == 0


def test_caps_lock_cycles_crops(tmp_path, qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from microalgae.ui.crop_review import CropReviewWindow
    monkeypatch.setitem(DETECTORS, "test", Substitute)
    root = tmp_path / "raw"
    root.mkdir()
    cv2.imwrite(str(root / "image.png"), np.full((12, 15, 3), 120, np.uint8))
    batch = CropBatch.prepare(root, tmp_path / "batch", dict(name="test"), files=list(root.glob("*")))
    batch.add(batch.data["images"][0], [7, 2, 10, 6])
    window = CropReviewWindow(batch)
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    window.images.setFocus()
    qtbot.waitUntil(window.isActiveWindow)
    qtbot.keyClick(window.images, Qt.Key.Key_CapsLock)
    assert window.crops.currentRow() == 1
    assert window.crop == window.active[1]
    qtbot.keyClick(window.images, Qt.Key.Key_CapsLock)
    assert window.crops.currentRow() == 0
    window.crops.clear()
    window.next_crop()
    assert window.crops.currentRow() == -1


def test_discard_persistence_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setitem(DETECTORS, "test", Substitute)
    root = tmp_path/"raw"
    root.mkdir()
    cv2.imwrite(str(root/"x.png"), np.full((12,15,3),120,np.uint8))
    batch = CropBatch.prepare(root,tmp_path/"batch",dict(name="test"),files=list(root.glob("*")))
    row = batch.data['images'][0]
    old = row['crops'][0]
    replacement = batch.add(row,[2,2,6,6],replaces=old['crop_id'])
    batch.save()
    stale = CropBatch(batch.directory)
    batch.discard(row,old)
    assert replacement['replaces'] is None
    assert not (batch.directory/old['path']).exists()
    with pytest.raises(ValueError):
        stale.discard(stale.data['images'][0],stale.data['images'][0]['crops'][0])
    original_unlink = Path.unlink
    def fail(path, *args, **kwargs):
        if path.name == Path(replacement['path']).name:
            raise PermissionError('locked file')
        return original_unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',fail)
    with pytest.raises(PermissionError):
        batch.discard(row,replacement)
    assert batch.data['images'][0]['crops'] == []
    saved_review = batch.path.read_bytes()
    pending_crop = batch.directory / replacement['path']
    saved_crop = pending_crop.read_bytes()
    CropBatch.audit(batch.directory)
    assert batch.path.read_bytes() == saved_review
    assert pending_crop.read_bytes() == saved_crop
    with (batch.directory / 'crop_manifest.csv').open(encoding='utf-8', newline='') as stream:
        assert list(csv.DictReader(stream)) == []
    monkeypatch.setattr(Path,'unlink',original_unlink)
    reopened = CropBatch(batch.directory)
    assert 'pending_crop_deletions' not in reopened.data
    assert not (batch.directory/replacement['path']).exists()
    reopened.complete(reopened.data['images'][0])
    reopened.save()


def test_discard_shortcut_selection(tmp_path, qtbot, monkeypatch):
    from threading import Event
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QPushButton
    from microalgae.ui.crop_review import CropReviewWindow
    monkeypatch.setitem(DETECTORS,'test',Substitute)
    root = tmp_path/'raw'
    root.mkdir()
    cv2.imwrite(str(root/'x.png'),np.full((12,15,3),120,np.uint8))
    batch = CropBatch.prepare(root,tmp_path/'batch',dict(name='test'),files=list(root.glob('*')))
    row = batch.data['images'][0]
    second = batch.add(row,[6,1,10,5])
    third = batch.add(row,[7,5,11,9])
    batch.save()
    window = CropReviewWindow(batch)
    qtbot.addWidget(window)
    window.show()
    window.activateWindow()
    window.crops.setFocus()
    qtbot.waitUntil(window.isActiveWindow)
    started = Event()
    release = Event()
    discard = batch.discard

    def delayed_discard(row, crop):
        started.set()
        release.wait(timeout=5)
        discard(row, crop)

    monkeypatch.setattr(batch, 'discard', delayed_discard)
    try:
        qtbot.keyClick(window.crops, Qt.Key.Key_R)
        qtbot.waitUntil(started.is_set)
        progress = QApplication.activeModalWidget()
        assert progress is not None
        qtbot.keyClick(progress, Qt.Key.Key_Escape)
        assert progress.isVisible()
        assert window.crops.count() == 3
    finally:
        release.set()
    qtbot.waitUntil(lambda: window.crop is second)
    assert window.crop['crop_id'] == second['crop_id']
    window.next_crop()
    assert window.crop['crop_id'] == third['crop_id']
    button = next(b for b in window.findChildren(QPushButton) if b.text()=='Discard crop (R)')
    button.click()
    qtbot.waitUntil(lambda: window.crop is second)
    assert window.crop['crop_id'] == second['crop_id']
    button.click()
    qtbot.waitUntil(lambda: window.crop is None)
    assert window.crop is None
    assert window.preview.pixmap().isNull()
    assert not window.viewer._overlay_items
    assert CropBatch(batch.directory).data['images'][0]['crops'] == []


def test_discard_invalid_shared_and_missing_paths(tmp_path, monkeypatch):
    monkeypatch.setitem(DETECTORS,'test',Substitute)
    root=tmp_path/'raw'
    root.mkdir()
    cv2.imwrite(str(root/'x.png'),np.full((12,15,3),120,np.uint8))
    batch=CropBatch.prepare(root,tmp_path/'batch',dict(name='test'),files=list(root.glob('*')))
    row=batch.data['images'][0]
    crop=row['crops'][0]
    path=crop['path']
    crop['path']='../outside.png'
    with pytest.raises(ValueError):batch.discard(row,crop)
    crop['path']=path
    second=batch.add(row,[2,2,5,5])
    second_path=second['path']
    second['path']=path
    with pytest.raises(ValueError):batch.discard(row,crop)
    second['path']='crops/../' + path
    with pytest.raises(ValueError):batch.discard(row,crop)
    second['path']=second_path
    (batch.directory/path).unlink()
    batch.discard(row,crop)
    assert row['crops']==[second]


def test_crop_edits_preserve_zoom_and_pan(tmp_path, qtbot, monkeypatch):
    from microalgae.ui.crop_review import CropReviewWindow
    monkeypatch.setitem(DETECTORS, 'test', Substitute)
    root = tmp_path / 'raw'
    root.mkdir()
    for name in ['first.png', 'second.png']:
        cv2.imwrite(str(root / name), np.full((1000, 1200, 3), 120, np.uint8))
    batch = CropBatch.prepare(root, tmp_path / 'batch', dict(name='test'), files=list(root.glob('*')))
    window = CropReviewWindow(batch)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)
    viewer = window.viewer
    viewer.zoom(3)
    viewer.horizontalScrollBar().setValue(150)
    viewer.verticalScrollBar().setValue(180)
    transform = viewer.transform()
    position = viewer.mapToScene(viewer.viewport().rect().center())
    for replace in [False, True]:
        window.draw(replace)
        window.add_box([100, 100, 200, 200])
        assert viewer.transform() == transform
        assert (viewer.mapToScene(viewer.viewport().rect().center()) - position).manhattanLength() < 1
        assert not viewer._fit_on_resize
    window.images.setCurrentRow(1)
    assert viewer._fit_on_resize
    assert viewer.transform().m11() < transform.m11()
