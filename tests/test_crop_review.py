import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from microalgae.imaging.crop_review import CropBatch, DEFAULT, DETECTORS, read_image

class Substitute:
    def __init__(self,settings): pass
    def detect(self,image):
        return [dict(box=[1,1,5,5],crop_box=[0,0,7,7])]


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
    batch=CropBatch(root/"crop_review_20261115_01")
    rows={r["image_id"]:r for r in batch.data["images"]}
    count=0
    for stage,candidate in [("development",5),("evaluation",0)]:
        directory=root/"detection_rerun_20261115_08"/stage
        for prediction in json.loads((directory/"predictions.json").read_text()):
            if prediction["candidate"]!=candidate:continue
            row=rows[prediction["image_id"]]
            assert [(c["box"],c["crop_box"]) for c in row["crops"]]==[(c["box"],c["crop_box"]) for c in prediction["regions"]]
            for i,crop in enumerate(row["crops"]):
                path=directory/"crops"/Path(row["image_id"]).parent/(Path(row["image_id"]).name+f"__candidate_{candidate:02d}__crop_{i:03d}.png")
                assert np.array_equal(read_image(path),read_image(batch.directory/crop["path"]))
            count+=1
    assert count==30


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
    assert window.review_counter.text() == "0/1 reviewed"
    assert window.crop["box"]==[1,1,5,5]
    window.decide("Non-target alga");window.complete()
    assert window.review_counter.text() == "1/1 reviewed"
    assert CropBatch(batch.directory).data["images"][0]["reviewed"]
    reopened = CropReviewWindow(CropBatch(batch.directory));qtbot.addWidget(reopened)
    assert reopened.review_counter.text() == "1/1 reviewed"
    reopened.decide("Unsure / mixed")
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
    for crop in row["crops"]:batch.decide(row,crop,"Reject artifact")
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
