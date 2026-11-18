"""Independent artifact recount; never imports the pilot scorer."""
import ast
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import cv2
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scripts._detection_references import DetectionSession,digest,read_image


def verify(references, run, original_session='results/detection_reference_01', *, frozen_only=False):
    s=DetectionSession(references);run=Path(run)
    split=json.loads((Path(references)/'split.json').read_text())
    assert digest(s.path)==split['snapshot_sha256']
    if not frozen_only:
        original=DetectionSession(original_session)
        assert digest(original.path)==split['original_annotation_sha256']
        clean = lambda data: {r['image_id']: {k:v for k,v in r.items() if k!='partition'} for r in data['images']}
        assert clean(original.data)==clean(s.data)
    rows={r['image_id']:r for r in s.data['images']}
    assignments = {a['image_id']: a for a in split['assignments']}
    assert len(assignments) == len(split['assignments']) == len(rows)
    assert set(assignments) == set(rows)
    partitions = {}
    for identifier, row in rows.items():
        assignment = assignments[identifier]
        assert assignment['partition'] == row['partition']
        partitions.setdefault(assignment['group_id'], set()).add(row['partition'])
    assert all(len(p) == 1 for p in partitions.values())
    if split.get('grouping_method') == 'verified_parent_specimen':
        path = Path(references)/'sources.csv'
        assert digest(path) == split['source_manifest_sha256']
        with path.open(newline='', encoding='utf-8-sig') as stream:
            entries = list(csv.DictReader(stream))
        sources = {r['image_id'].strip(): {k:v.strip() for k,v in r.items()} for r in entries}
        assert len(entries) == len(sources) == len(rows) and set(sources) == set(rows)
        assert all(r['evidence'] and (r['parent_image_id'] or r['specimen_id']) for r in sources.values())
        unseen = set(rows)
        while unseen:
            component = {min(unseen)}
            pending = list(component)
            while pending:
                a = pending.pop()
                for b in unseen-component:
                    linked = rows[a]['sha256'] == rows[b]['sha256'] or any(
                        sources[a][k] and sources[a][k] == sources[b][k]
                        for k in ('parent_image_id', 'specimen_id'))
                    if linked:
                        component.add(b)
                        pending.append(b)
            assert all(assignments[i]['group_id'] == min(component) for i in component)
            unseen -= component
    metadata=json.loads((run/'metadata.json').read_text())
    if 'split_sha256' in metadata:
        assert metadata['split_sha256'] == digest(Path(references)/'split.json')
    assert all(digest(Path(p))==h for p,h in metadata['code_hashes'].items())
    predictions={(p['candidate'],p['image_id']):p for p in json.loads((run/'predictions.json').read_text())}
    scores=json.loads((run/'scores.json').read_text())
    for score in scores:
        row=rows[score['image_id']];objects=[o for o in row['objects'] if not o['uncertain']]
        predictions_for_image=predictions[score['candidate'],score['image_id']]['regions']
        matrix=np.zeros((len(predictions_for_image),len(objects)))
        uncertain=np.zeros((row['height'],row['width']),bool)
        for o in row['objects']:
            if o['uncertain']:
                l,t,r,b=o['box'];uncertain[t:b,l:r]=True
        for i,p in enumerate(predictions_for_image):
            a=p['box']
            for j,o in enumerate(objects):
                b=o['box'];inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                matrix[i,j]=inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
        cardinality=int((maximum_bipartite_matching(csr_matrix(matrix>=score['threshold']),perm_type='column')>=0).sum()) if matrix.size else 0
        assert cardinality==score['tp']
        used=set();ref_ids=set()
        for m in score['matches']:
            j=next(j for j,o in enumerate(objects) if o['object_id']==m['reference'])
            assert m['prediction'] not in used and m['reference'] not in ref_ids
            used.add(m['prediction']);ref_ids.add(m['reference'])
            assert np.isclose(matrix[m['prediction'],j],m['iou']) and m['iou']>=score['threshold']
        ignored=[]
        for i,p in enumerate(predictions_for_image):
            l,t,r,b=p['box']
            if i not in used and not (matrix[i]>0).any() and uncertain[t:b,l:r].mean()>=.5:ignored.append(i)
        assert ignored==score['ignored_indices']
        tp=len(used);fp=len(predictions_for_image)-tp-len(ignored);fn=len(objects)-tp
        assert (tp,fp,fn)==(score['tp'],score['fp'],score['fn'])
        assert np.isclose(score['f1'],2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    rankings = pd.read_csv(run/'rankings.csv')
    if 'mean_folder_f1' in rankings:
        for ranking in rankings.itertuples():
            items = [r for r in scores if r['candidate']==ranking.candidate and r['threshold']==.5]
            folder_metrics = []
            for folder in sorted({r['source_folder'] for r in items}):
                totals = [sum(r[k] for r in items if r['source_folder']==folder) for k in ('tp','fp','fn')]
                tp,fp,fn = totals
                folder_metrics.append((2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0, tp/(tp+fn) if tp+fn else 0))
            assert np.isclose(ranking.mean_folder_f1, np.mean([m[0] for m in folder_metrics]))
            assert np.isclose(ranking.mean_folder_recall, np.mean([m[1] for m in folder_metrics]))
        if metadata.get('stage')=='develop':
            selection = json.loads((run/'selection.json').read_text())
            metric = metadata['settings'].get('selection_metric', 'f1')
            recall = 'mean_folder_recall' if metric=='mean_folder_f1' else 'recall'
            winner = rankings.sort_values([metric,recall,'low_ratio','background_kernel_size','min_component_area'], ascending=[False,False,True,True,True]).iloc[0]
            assert selection['selection_metric']==metric
            assert all(selection['selected'][k]==winner[k] for k in ('low_ratio','background_kernel_size','min_component_area'))
    coverage=pd.read_csv(run/'coverage.csv')
    assert coverage.image_id.is_unique
    assert len(scores)==len(coverage)*len(metadata['candidates'])*2
    assert len(predictions)==len(coverage)*len(metadata['candidates'])
    expected=set(coverage.image_id)
    if 'stage' in metadata:
        partition = 'development' if metadata['stage']=='develop' else 'evaluation'
        assert expected == {i for i,r in rows.items() if r['partition']==partition}
    assert all({key[1] for key in predictions if key[0]==i}==expected for i in range(len(metadata['candidates'])))
    crops = pd.read_csv(run/'crop_manifest.csv')
    assert crops.path.is_unique
    lineage = {'parent_image_id', 'crop_image_id', 'partition'}
    if lineage.intersection(crops.columns):
        assert lineage.issubset(crops.columns)
        assert crops.crop_image_id.is_unique
        for r in crops.itertuples():
            assert r.parent_image_id == r.image_id
            assert r.crop_image_id == Path(r.path).as_posix()
            assert r.partition == rows[r.parent_image_id]['partition']
            regions = predictions[r.candidate, r.parent_image_id]['regions']
            assert ast.literal_eval(r.crop_box) == regions[r.prediction]['crop_box']
    for r in crops.itertuples():
        box=ast.literal_eval(r.crop_box);l,t,right,b=box
        raw=read_image(s.root/r.image_id)
        assert np.array_equal(read_image(run/r.path),raw[t:b,l:right])
    review = pd.read_csv(run/'visual_review.csv')
    assert set(review.status)=={'unreviewed'}
    if metadata['settings'].get('review_all_candidates', False):
        assert not review.duplicated(['image_id','candidate']).any()
        assert set(zip(review.candidate, review.image_id)) == set(predictions)
        assert set(zip(crops.candidate,crops.image_id,crops.prediction)) == {
            (candidate,identifier,i) for (candidate,identifier),p in predictions.items() for i in range(len(p['regions']))}

    receipt=dict(score_rows_recounted=len(scores),all_source_and_annotation_hashes_unchanged=not frozen_only,
                 frozen_snapshot_and_source_hashes_verified=True,original_annotation_verified=not frozen_only,
                 coverage_verified=True,crops_pixel_identical=True,manual_review='pending')
    (run/'independent_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(run,receipt)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('references', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--original-session', type=Path, default=Path('results/detection_reference_01'))
    parser.add_argument('--frozen-only', action='store_true', help='Verify frozen artifacts without claiming verification of the original annotation session')
    args = parser.parse_args()
    verify(args.references, args.run, args.original_session, frozen_only=args.frozen_only)
