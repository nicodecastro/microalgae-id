"""Reviewed references use original source pixels and both fixed baselines."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import scripts.run_detection_pilot as pilot
from microalgae.imaging.crop_review import digest


@pytest.fixture
def review(tmp_path):
    root = tmp_path/'raw'
    root.mkdir()
    batch = tmp_path/'review'
    batch.mkdir()
    image = np.full((30,30,3),120,np.uint8)
    cv2.imwrite(str(root/'image.png'),image)
    local = batch/'sources'
    local.mkdir()
    ok, encoded = cv2.imencode('.png',image)
    encoded.tofile(local/'alga_é.png')
    def crop(identifier, box, decision, active=True):
        return dict(crop_id=identifier,box=box,decision=decision,active=active)
    row = dict(image_id='n_fruticosa/image.png',target='n_fruticosa',reviewed=True,
               status='proposals',source_path='image.png',sha256=digest(root/'image.png'),
               crops=[crop('a',[0,0,10,10],'Accept target'),
                      crop('b',[12,0,20,10],'Non-target alga'),
                      crop('c',[0,15,10,25],'Unsure / mixed'),
                      crop('d',[0,0,30,30],'Accept target',False),
                      crop('e',[20,20,25,25],'Reject artifact')])
    empty = dict(row,image_id='n_fruticosa/empty.png',source_path='sources/alga_é.png',
                 batch_local_source=True,crops=[],status='zero proposals')
    pending = dict(row,image_id='n_fruticosa/pending.png',reviewed=False)
    other = dict(row,image_id='n_palea/image.png',target='n_palea')
    data = dict(schema_version=1,root=str(root),images=[row,empty,pending,other],pending_crop_deletions=['crops/missing.png'])
    path=batch/'review.json'
    path.write_text(json.dumps(data),encoding='utf-8')
    return batch,path,data


def test_review_scoring_outputs(review,tmp_path,monkeypatch):
    batch,path,data=review
    before=path.read_bytes()
    def detect(image, **settings):
        boxes=[[0,0,10,10],[12,0,20,10],[0,15,10,25]]
        return dict(regions=[dict(box=b,crop_box=b,component_area=80,truncated=True) for b in boxes])
    monkeypatch.setattr(pilot,'detect_regions',detect)
    output=pilot.score_review(batch,tmp_path/'scores')
    assert path.read_bytes()==before
    coverage=pd.read_csv(output/'coverage.csv')
    assert len(coverage)==4 and coverage.included.sum()==3
    rankings=pd.read_csv(output/'rankings.csv')
    assert set(coverage.image_id)=={r['image_id'] for r in data['images']}
    assert list(rankings.name)==['control','h65']
    assert list(rankings.tp)==[4,4]
    assert list(rankings.fp)==[3,3]
    assert list(rankings.fn)==[0,0]
    assert len(list((output/'galleries').glob('*.png')))==6
    assert (output/'review.md').is_file()
    assert (output/'review_snapshot.json').read_bytes()==before
    assert json.loads((output/'verification.json').read_text())['crops_pixel_verified']==18
    refs=json.loads((output/'references.json').read_text())
    assert [o['object_id'] for o in refs[0]['objects']]==['a','b','c']
    assert refs[0]['objects'][1]['species']=='unknown'
    with pytest.raises(FileExistsError):pilot.score_review(batch,output)


def test_review_input_validation(review,tmp_path):
    batch,path,data=review
    pending = json.loads(json.dumps(data))
    for row in pending['images']:
        row['reviewed'] = False
    path.write_text(json.dumps(pending))
    with pytest.raises(ValueError,match='No reviewed'):pilot.score_review(batch,tmp_path/'none')
    path.write_text(json.dumps(data))
    assert not (tmp_path/'none').exists()
    data['images'][0]['crops'][0]['box']=[0,0,100,100]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='coordinates'):pilot.score_review(batch,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    data['images'][0]['crops'][0]['box']=[0,0,10,10]
    data['images'][0]['sha256']='changed'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Source image changed'):pilot.review_references(batch)


def test_review_changed_during_capture(review,monkeypatch):
    batch,path,data=review
    original=pilot.ReviewReferences.verify_source
    def change(self,row):
        resolved=original(self,row)
        path.write_bytes(path.read_bytes()+b' ')
        return resolved
    monkeypatch.setattr(pilot.ReviewReferences,'verify_source',change)
    with pytest.raises(ValueError,match='Review changed'):pilot.review_references(batch)


def test_review_cli(review, tmp_path, monkeypatch):
    import sys
    batch, path, data = review
    output = tmp_path/'cli'
    monkeypatch.setattr(sys,'argv',['run_detection_pilot.py','score-review',
        '--crop-review',str(batch),'--output-dir',str(output)])
    pilot.main()
    assert json.loads((output/'run_status.json').read_text())['status']=='completed'
    assert list(pd.read_csv(output/'rankings.csv').name)==['control','h65']
