from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from scripts.run_detection_pilot import score_boxes, family, fresh
from microalgae.imaging.detection import detect_regions


def reference(box, identifier='a', uncertain=False):
    return dict(box=box,object_id=identifier,uncertain=uncertain)


def test_duplicates_empty_and_ignore_protection():
    row=dict(objects=[reference([0,0,10,10]),reference([0,0,30,30],'ignore',True)])
    s=score_boxes([dict(box=[0,0,10,10])]*2,row)
    assert (s['tp'],s['fp'],s['fn'],s['ignored'])==(1,1,0,0)
    assert score_boxes([],row)['fn']==1
    assert score_boxes([dict(box=[15,15,20,20])],row)['ignored']==1


def test_uncertain_union_does_not_double_count():
    row=dict(objects=[reference([0,0,4,10],'a',True),reference([0,0,4,10],'b',True)])
    assert score_boxes([dict(box=[0,0,10,10])],row)['fp']==1


def test_global_matching_and_threshold():
    row=dict(objects=[reference([0,0,10,10]),reference([5,0,15,10],'b')])
    s=score_boxes([dict(box=[0,0,15,10]),dict(box=[0,0,10,10])],row)
    assert s['tp']==2
    assert score_boxes([dict(box=[0,0,15,10])],row,.75)['tp']==0


def test_coordinates_and_clipping(monkeypatch):
    import microalgae.imaging.detection as module
    def fake(gray,*args,**kwargs):
        mask=np.zeros(gray.shape,bool);mask[0:10,0:10]=True;return mask
    monkeypatch.setattr(module,'geometry_foreground_mask',fake)
    result=detect_regions(np.zeros((1024,2048,3),np.uint8),low_ratio=.5,background_kernel_size=15,min_component_area=4)
    assert result['transform']['working_size']==[512,256]
    assert result['regions'][0]['box']==[0,0,40,40]
    assert result['regions'][0]['crop_box']==[0,0,44,44]
    assert result['regions'][0]['truncated']


def test_groups_and_no_overwrite(tmp_path):
    a=dict(image_id='n/image_1.jpg',source_folder='n')
    b=dict(image_id='n/image_2.jpg',source_folder='n')
    assert family(a)==family(b)
    with pytest.raises(FileExistsError):fresh(tmp_path)


def test_development_isolation_lock_and_failed_run(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import scripts.run_detection_pilot as pilot
    refs=tmp_path/'refs';refs.mkdir()
    path=refs/'annotations.json';path.write_text('{}')
    rows=[dict(image_id=str(i),source_folder='a',partition='development' if i<20 else 'evaluation',objects=[]) for i in range(30)]
    session=SimpleNamespace(path=path,root=tmp_path/'raw',data={'images':rows},verify_source=lambda row:int(row['image_id']))
    pilot.write(refs/'split.json',dict(snapshot_sha256=pilot.digest(path),assignments=[dict(image_id=r['image_id'],partition=r['partition'],group_id=r['image_id']) for r in rows]))
    monkeypatch.setattr(pilot,'DetectionSession',lambda p:session)
    monkeypatch.setattr(pilot,'read_image',lambda i:np.full((2,2,3),i,np.uint8))
    seen=[]
    def detector(image,**kwargs):
        seen.append(int(image[0,0,0]));return dict(regions=[],transform={})
    monkeypatch.setattr(pilot,'detect_regions',detector)
    monkeypatch.setattr(pilot,'galleries',lambda *args:None)
    config=tmp_path/'settings.yaml'
    config.write_text('max_side: 512\nmargin_ratio: 0.1\nlow_ratios: [0.5,0.65]\nbackground_kernels: [15,31,61]\nminimum_areas: [4,25,64]\n')
    dev=pilot.run('develop',refs,tmp_path/'dev',config=config)
    assert len(seen)==360 and set(seen)==set(range(20))
    seen.clear()
    pilot.run('evaluate',refs,tmp_path/'eval',development=dev)
    assert len(seen)==30 and set(seen)==set(range(20,30))
    lock=dev/'selection.json';before=lock.read_bytes()
    assert json.loads(before)['selected']['low_ratio']==.5
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='snapshot changed'):
        pilot.run('evaluate',refs,tmp_path/'changed',development=dev)
    assert not (tmp_path/'changed').exists() and lock.read_bytes()==before
    path.write_text('{}')
    def fail(*a,**kw):raise OSError('read failure')
    monkeypatch.setattr(pilot,'detect_regions',fail)
    with pytest.raises(OSError):pilot.run('develop',refs,tmp_path/'failed',config=config)
    assert json.loads((tmp_path/'failed/run_status.json').read_text())['status']=='failed'
    assert not (tmp_path/'failed/selection.json').exists()


def test_verified_groups_transitive_and_cross_folder():
    from scripts.run_detection_pilot import source_groups
    rows = [dict(image_id=str(i), source_folder=str(i), sha256=str(i)) for i in range(4)]
    sources = {str(i): dict(parent_image_id='', specimen_id='') for i in range(4)}
    sources['0']['parent_image_id'] = sources['1']['parent_image_id'] = 'parent'
    sources['1']['specimen_id'] = sources['2']['specimen_id'] = 'specimen'
    rows[3]['sha256'] = rows[2]['sha256']
    assert source_groups(rows, sources) == [[0, 1, 2, 3]]


@pytest.fixture
def source_pilot(tmp_path):
    import csv
    import cv2
    from scripts._detection_references import DetectionSession
    raw = tmp_path/'raw'
    identifiers = []
    entries = []
    for folder in ('m_aeruginosa', 'n_fruticosa', 'n_palea'):
        (raw/folder).mkdir(parents=True)
        for i in range(10):
            identifier = f'{folder}/{i}.png'
            cv2.imwrite(str(raw/identifier), np.full((12, 12, 3), len(identifiers), np.uint8))
            identifiers.append(identifier)
            group = 0 if i < 3 else 1 if i < 6 else 2
            entries.append(dict(image_id=identifier, parent_image_id=f'{folder}/{group}',
                                specimen_id='', evidence='synthetic acquisition log'))
    session = DetectionSession.create(raw, tmp_path/'session', image_ids=identifiers)
    for i, row in enumerate(deepcopy(session.data['images'])):
        row.update(status='reviewed', objects=[dict(object_id='a', box=[1,1,5,5], species='unknown',
                   kind='cell', uncertain=False, truncated=False, notes='')])
        session.save(i, row)
    manifest = tmp_path/'sources.csv'
    def save(entries):
        with manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_id','parent_image_id','specimen_id','evidence'])
            writer.writeheader(); writer.writerows(entries)
    save(entries)
    return session, manifest, entries, save


def test_source_prepare_reproducible_and_validation(source_pilot, tmp_path):
    import json
    from scripts.run_detection_pilot import prepare, source_manifest, validate_sources
    session, manifest, entries, save = source_pilot
    first = prepare(session.directory, tmp_path/'first', manifest)
    split = json.loads((first/'split.json').read_text())
    save(list(reversed(entries)))
    second = prepare(session.directory, tmp_path/'second', manifest)
    assert split['assignments'] == json.loads((second/'split.json').read_text())['assignments']
    assert sum(a['partition']=='evaluation' for a in split['assignments']) == 10
    for invalid in (entries[:-1], entries+[entries[0]], [dict(e, evidence='') for e in entries],
                    [dict(e, parent_image_id='', specimen_id='') for e in entries],
                    [dict(e, image_id='unknown') if i==0 else e for i,e in enumerate(entries)]):
        save(invalid)
        with pytest.raises(ValueError): source_manifest(manifest, session.data['images'])
    save([dict(e, parent_image_id='one') for e in entries])
    with pytest.raises(ValueError, match='No feasible grouped split'):
        prepare(session.directory, tmp_path/'impossible', manifest)
    assert not (tmp_path/'impossible').exists()
    rows = json.loads((first/'annotations.json').read_text())['images']
    changed = deepcopy(split); changed['assignments'][0]['group_id'] = 'changed'
    with pytest.raises(ValueError, match='assignments changed'): validate_sources(first, changed, rows)
    with (first/'sources.csv').open('a') as stream: stream.write('tampered')
    with pytest.raises(ValueError, match='manifest changed'): validate_sources(first, split, rows)


def test_source_pilot_end_to_end(source_pilot, tmp_path, monkeypatch):
    import json
    import scripts.run_detection_pilot as pilot
    from scripts.verify_detection_pilot import verify
    session, manifest, entries, save = source_pilot
    original = session.path.read_bytes()
    refs = pilot.prepare(session.directory, tmp_path/'refs', manifest)
    monkeypatch.setattr(pilot, 'detect_regions', lambda image, **kw: dict(
        regions=[dict(box=[1,1,5,5], crop_box=[1,1,5,5], truncated=False)], transform={}))
    config = Path('configs/detection_pilot.yaml')
    development = pilot.run('develop', refs, tmp_path/'dev', config=config)
    evaluation = pilot.run('evaluate', refs, tmp_path/'eval', development=development)
    verify(refs, development, session.directory)
    verify(refs, evaluation, session.directory)
    import pandas as pd
    for run_dir in (development, evaluation):
        crops = pd.read_csv(run_dir/'crop_manifest.csv')
        assert crops.crop_image_id.is_unique
        assert (crops.parent_image_id == crops.image_id).all()
        assert (crops.crop_image_id == crops.path).all()
        for crop in crops.itertuples():
            assert Path(crop.path).name == f'{Path(crop.image_id).name}__candidate_{crop.candidate:02d}__crop_{crop.prediction:03d}.png'
            assert (run_dir/crop.path).is_file()
        review = pd.read_csv(run_dir/'visual_review.csv')
        expected_candidates = 12 if run_dir==development else 3
        assert review.candidate.nunique() == expected_candidates
        assert len(review) == (20 if run_dir==development else 10)*expected_candidates
        assert set(crops.partition) == {'development' if run_dir==development else 'evaluation'}
    assert session.path.read_bytes() == original
    assert json.loads((evaluation/'independent_verification.json').read_text())['coverage_verified']
    changed_original = json.loads(original)
    changed_original['images'][0]['notes'] = 'Edited after freezing'
    session.path.write_text(json.dumps(changed_original))
    with pytest.raises(AssertionError): verify(refs, evaluation, session.directory)
    verify(refs, evaluation, session.directory, frozen_only=True)
    receipt = json.loads((evaluation/'independent_verification.json').read_text())
    assert receipt['frozen_snapshot_and_source_hashes_verified']
    assert not receipt['original_annotation_verified']
    assert not receipt['all_source_and_annotation_hashes_unchanged']
    session.path.write_bytes(original)
    crop_path = evaluation/'crop_manifest.csv'
    saved_crops = crop_path.read_bytes()
    changed_crops = pd.read_csv(crop_path)
    changed_crops.loc[0, 'parent_image_id'] = 'wrong-parent'
    changed_crops.to_csv(crop_path, index=False)
    with pytest.raises(AssertionError): verify(refs, evaluation, session.directory)
    crop_path.write_bytes(saved_crops)
    split_path = refs/'split.json'
    before = split_path.read_bytes()
    split = json.loads(before)
    split['assignments'][0]['group_id'] = 'invalid'
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match='assignments changed'):
        pilot.run('develop', refs, tmp_path/'bad_group', config=config)
    with pytest.raises(AssertionError): verify(refs, evaluation, session.directory)
    assert not (tmp_path/'bad_group').exists()
    split_path.write_bytes(before)
    with pytest.raises(FileExistsError): pilot.prepare(session.directory, refs, manifest)


def test_empty_crop_manifest_has_lineage_headers(tmp_path):
    import pandas as pd
    from scripts.run_detection_pilot import galleries
    galleries(tmp_path, None, [], [], [], [])
    crops = pd.read_csv(tmp_path/'crop_manifest.csv')
    assert crops.empty
    assert {'image_id', 'parent_image_id', 'crop_image_id', 'partition', 'path'}.issubset(crops.columns)


def test_development_grid_expansion():
    import yaml
    from scripts.run_detection_pilot import development_candidates
    settings = yaml.safe_load(Path('configs/detection_pilot.yaml').read_text())
    candidates = development_candidates(settings)
    assert len(candidates) == 12
    anchors = [(.4,15,128), (.4,31,128), (.5,15,128)]
    triples = [(c['low_ratio'], c['background_kernel_size'], c['min_component_area']) for c in candidates]
    assert triples[:3] == anchors
    assert set(triples) == set(anchors) | {
        neighbor for l,k,a in anchors for neighbor in ((round(l-.05,2),k,a),(l,k-4,a),(l,k,a+32))}

    for extra in ([candidates[0]], [dict(low_ratio=.5)], [dict(low_ratio=.5,background_kernel_size=14,min_component_area=96)],
                  [dict(low_ratio=float('nan'),background_kernel_size=15,min_component_area=96)],
                  [dict(low_ratio=.5,background_kernel_size=15,min_component_area=0)], 'invalid'):
        with pytest.raises(ValueError):
            development_candidates(dict(settings, extra_candidates=extra))


def test_equal_folder_selection_differs_from_pooled():
    from scripts.run_detection_pilot import ranking_scores, select_winner
    def candidate(i, counts):
        records = [dict(source_folder=str(j), tp=t, fp=p, fn=n) for j,(t,p,n) in enumerate(counts)]
        return dict(candidate=i, low_ratio=.5, background_kernel_size=15, min_component_area=64+i,
                    **ranking_scores(records))
    dominant = candidate(0, [(90,0,10),(0,0,5),(0,0,5)])
    balanced = candidate(1, [(50,0,50),(5,0,0),(5,0,0)])
    assert select_winner([dominant,balanced], 'f1')['candidate'] == 0
    assert select_winner([dominant,balanced], 'mean_folder_f1')['candidate'] == 1
    assert balanced['mean_folder_f1'] == pytest.approx((2/3+1+1)/3)
    assert select_winner([dict(balanced, min_component_area=200),balanced], 'mean_folder_f1')['min_component_area'] == 65
