"""Frozen development/evaluation pilot for class-agnostic region detection."""
import argparse
import csv
from collections import Counter, defaultdict
from copy import deepcopy
import itertools
import json
from pathlib import Path
import platform
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import cv2
import numpy as np
import pandas as pd
import yaml
import scipy
from scipy.optimize import linear_sum_assignment
from microalgae.imaging.detection import detect_regions
from scripts._detection_references import DetectionSession, digest, read_image


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def family(row):
    stem = Path(row['image_id']).stem
    reason = 'individual image; provenance unknown'
    if '.x_' in stem:
        stem = re.sub(r'\.Chunk\d+', '', stem.split('.x_')[0]); reason = 'microscopy parent prefix'
    elif re.fullmatch(r'.*img\d+_\d+_[0-9a-f]+', stem):
        stem = re.sub(r'_\d+_[0-9a-f]+$', '', stem); reason = 'img parent prefix'
    elif re.fullmatch(r'\d+', stem):
        stem = 'numeric_names'; reason = 'conservative numbered family'
    elif re.fullmatch(r'image_\d+', stem):
        stem = 'image_number_names'; reason = 'conservative image_N family'
    elif re.fullmatch(r'nfruticosa_or_nactinastroides\d+', stem):
        stem = 'nfruticosa_or_nactinastroides'; reason = 'conservative named family'
    return row['source_folder']+'/'+stem, reason


def source_manifest(path, rows):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'image_id', 'parent_image_id', 'specimen_id', 'evidence'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Source manifest needs image_id, parent_image_id, specimen_id, evidence')
        entries = list(reader)
    mapping = {}
    for entry in entries:
        if any(not isinstance(entry.get(k), str) for k in required):
            raise ValueError('Invalid source manifest row')
        entry = {k: entry[k].strip() for k in required}
        if not entry['image_id'] or entry['image_id'] in mapping:
            raise ValueError('Duplicate or blank source manifest image_id')
        if not entry['evidence'] or not (entry['parent_image_id'] or entry['specimen_id']):
            raise ValueError('Every image needs documented parent/specimen identity')
        mapping[entry['image_id']] = entry
    if set(mapping) != {r['image_id'] for r in rows}:
        raise ValueError('Source manifest must cover exactly the reviewed images')
    return mapping


def source_groups(rows, sources):
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i
    for i, a in enumerate(rows):
        for j, b in enumerate(rows[:i]):
            linked = a['sha256'] == b['sha256']
            if sources is None:
                linked |= family(a)[0] == family(b)[0]
            else:
                x, y = sources[a['image_id']], sources[b['image_id']]
                linked |= any(x[k] and x[k] == y[k] for k in ('parent_image_id', 'specimen_id'))
            if linked:
                parents[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return list(groups.values())


def validate_sources(directory, split, rows):
    if split.get('grouping_method') != 'verified_parent_specimen':
        return
    path = Path(directory)/'sources.csv'
    if digest(path) != split['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    sources = source_manifest(path, rows)
    assignments = {a['image_id']: a for a in split['assignments']}
    for group in source_groups(rows, sources):
        expected = min(rows[i]['image_id'] for i in group)
        if any(assignments[rows[i]['image_id']]['group_id'] != expected for i in group):
            raise ValueError('Source group assignments changed')


def prepare(session_dir, output, manifest=None):
    session = DetectionSession(session_dir)
    if Path(output).resolve().is_relative_to(session.root):
        raise ValueError('Output cannot be inside source data')
    rows = session.data['images']
    if len(rows) != 30 or any(r['status'] != 'reviewed' for r in rows):
        raise ValueError('Expected 30 reviewed images')
    sources = source_manifest(manifest, rows) if manifest else None
    if sources is not None:
        rows = sorted(rows, key=lambda r: r['image_id'])
    groups = source_groups(rows, sources)
    targets = {'m_aeruginosa': 3, 'n_fruticosa': 3, 'n_palea': 4}
    if set(r['source_folder'] for r in rows) != set(targets):
        raise ValueError('Expected the three pilot source folders')
    def candidates(position=0, indices=(), counts=None):
        counts = counts or Counter()
        if any(counts[k] > v for k, v in targets.items()):
            return
        if position == len(groups):
            if all(counts[k] == v for k, v in targets.items()):
                yield indices
            return
        remaining = Counter(rows[i]['source_folder'] for g in groups[position:] for i in g)
        if any(counts[k]+remaining[k] < v for k,v in targets.items()):
            return
        yield from candidates(position+1, indices, counts)
        group = groups[position]
        yield from candidates(position+1, indices+tuple(group), counts+Counter(rows[i]['source_folder'] for i in group))
    def counts(indices):
        c = Counter()
        for i in indices:
            for obj in rows[i]['objects']:
                if not obj['uncertain']:
                    c.update(['species:'+obj['species'], 'kind:'+obj['kind'], 'clipped:'+str(obj['truncated'])])
        return c
    # Preserve historical candidate ordering and numeric IDs without a manifest.
    if sources is None:
        choices = []
        for folder, target in targets.items():
            local = [g for g in groups if rows[g[0]]['source_folder'] == folder]
            if any(len({rows[i]['source_folder'] for i in g}) != 1 for g in local):
                raise ValueError('Cross-folder duplicate groups require explicit split resolution')
            choices.append([sum((local[i] for i in indices), []) for n in range(len(local)+1)
                            for indices in itertools.combinations(range(len(local)), n)
                            if sum(len(local[i]) for i in indices) == target])
        options = (sum(candidate, []) for candidate in itertools.product(*choices))
    else:
        options = candidates()
    total = counts(range(len(rows)))
    best, selected = None, None
    rng = random.Random(42)
    for indices in options:
        c = counts(indices)
        key = (sum((c[k]/v-1/3)**2 for k,v in total.items()), rng.random())
        if best is None or key < best:
            best, selected = key, set(indices)
    if selected is None:
        detail = [[rows[i]['image_id'] for i in g] for g in groups]
        raise ValueError(f'No feasible grouped split; groups (sizes from membership): {detail}')
    group_ids = {i: min(rows[j]['image_id'] for j in g) if sources is not None else str(min(g))
                 for g in groups for i in g}
    snapshot = deepcopy(session.data)
    snapshot['images'] = deepcopy(rows)
    assignments = []
    for i, row in enumerate(snapshot['images']):
        row['partition'] = 'evaluation' if i in selected else 'development'
        assignments.append(dict(image_id=row['image_id'], group_id=group_ids[i],
                                reason='documented parent/specimen or exact duplicate' if sources is not None else family(row)[1],
                                partition=row['partition']))
    out = fresh(output)
    write(out/'annotations.json', snapshot)
    metadata = dict(original_annotation_sha256=digest(session.path), assignments=assignments,
                    snapshot_sha256=digest(out/'annotations.json'), seed=42, balance_score=best[0],
                    grouping_method='verified_parent_specimen' if sources is not None else 'inferred_filename_duplicate',
                    group_counts={p: len({a['group_id'] for a in assignments if a['partition']==p})
                                  for p in ('development', 'evaluation')})
    if sources is not None:
        (out/'sources.csv').write_bytes(Path(manifest).read_bytes())
        metadata['source_manifest_sha256'] = digest(out/'sources.csv')
        metadata['identity_coverage'] = {k: sum(bool(s[k]) for s in sources.values())
                                         for k in ('parent_image_id', 'specimen_id')}
    write(out/'split.json', metadata)
    DetectionSession(out).export(out/'snapshot')
    return out


def intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def area(a): return (a[2]-a[0])*(a[3]-a[1])


def score_boxes(predictions, row, threshold=0.5):
    boxes = [o['box'] for o in predictions]
    certain = [o for o in row['objects'] if not o['uncertain']]
    uncertain = [o['box'] for o in row['objects'] if o['uncertain']]
    matrix = np.zeros((len(boxes),len(certain)))
    for i,a in enumerate(boxes):
        for j,o in enumerate(certain):
            inter = intersection(a,o['box'])
            matrix[i,j] = inter/(area(a)+area(o['box'])-inter)
    matches=[]
    if matrix.size:
        reward = (matrix >= threshold)*(1+matrix/(min(matrix.shape)+1))
        ii,jj = linear_sum_assignment(-reward)
        matches=[dict(prediction=int(i), reference=certain[j]['object_id'], iou=float(matrix[i,j]))
                 for i,j in zip(ii,jj) if matrix[i,j] >= threshold]
    used={m['prediction'] for m in matches}
    ignored=[]
    for i,box in enumerate(boxes):
        if i in used or any(intersection(box,o['box']) for o in certain): continue
        l,t,r,b=box
        union=np.zeros((b-t,r-l),bool)
        for q in uncertain:
            if intersection(box,q): union[max(t,q[1])-t:min(b,q[3])-t,max(l,q[0])-l:min(r,q[2])-l]=True
        if union.sum()/area(box)>=.5: ignored.append(i)
    return dict(tp=len(matches),fp=len(boxes)-len(matches)-len(ignored),fn=len(certain)-len(matches),
                ignored=len(ignored),matches=matches,ignored_indices=ignored,
                possible_merges=sum(sum(intersection(box,o['box'])>0 for o in certain)>1 for i,box in enumerate(boxes) if i not in used),
                possible_splits=sum(sum(intersection(box,o['box'])>0 for i,box in enumerate(boxes) if i not in used)>1 for o in certain if o['object_id'] not in {m['reference'] for m in matches}))


def metrics(tp,fp,fn):
    return dict(precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)


def code_hashes():
    return {str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/microalgae/imaging/detection.py',ROOT/'src/microalgae/imaging/preprocessing/geometry_foreground.py',ROOT/'scripts/_detection_references.py']}


def development_candidates(settings):
    candidates = [dict(low_ratio=l, background_kernel_size=k, min_component_area=a)
                  for l,k,a in itertools.product(settings['low_ratios'], settings['background_kernels'], settings['minimum_areas'])]
    extras = settings.get('extra_candidates', [])
    if not isinstance(extras, list):
        raise ValueError('extra_candidates must be a list')
    candidates += extras
    if not candidates:
        raise ValueError('Expected at least one development configuration')
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {'low_ratio', 'background_kernel_size', 'min_component_area'}:
            raise ValueError('Invalid development configuration keys')
        low, kernel, area = (candidate[k] for k in ('low_ratio', 'background_kernel_size', 'min_component_area'))
        if type(low) not in (int, float) or not np.isfinite(low) or not 0 < low <= 1:
            raise ValueError('low_ratio must be finite and in (0, 1]')
        if type(kernel) is not int or kernel < 3 or kernel % 2 == 0:
            raise ValueError('background_kernel_size must be an odd integer >= 3')
        if type(area) is not int or area < 1:
            raise ValueError('min_component_area must be a positive integer')
        key = (low, kernel, area)
        if key in seen:
            raise ValueError('Duplicate development configuration')
        seen.add(key)
    return candidates


def ranking_scores(records):
    totals = {k: sum(r[k] for r in records) for k in ('tp', 'fp', 'fn')}
    folders = sorted({r['source_folder'] for r in records})
    per_folder = [metrics(**{k: sum(r[k] for r in records if r['source_folder']==folder)
                            for k in ('tp', 'fp', 'fn')}) for folder in folders]
    return dict(**totals, **metrics(**totals),
                mean_folder_f1=sum(m['f1'] for m in per_folder)/len(folders),
                mean_folder_recall=sum(m['recall'] for m in per_folder)/len(folders))


def select_winner(rankings, metric):
    recall = 'mean_folder_recall' if metric=='mean_folder_f1' else 'recall'
    return min(rankings, key=lambda r: (-r[metric], -r[recall], r['low_ratio'],
                                       r['background_kernel_size'], r['min_component_area']))


def run(stage, references, output, config=None, development=None):
    s=DetectionSession(references)
    if Path(output).resolve().is_relative_to(s.root): raise ValueError('Output cannot be inside source data')
    split=json.loads((Path(references)/'split.json').read_text())
    if digest(s.path)!=split['snapshot_sha256']: raise ValueError('Frozen snapshot changed')
    validate_sources(references, split, s.data['images'])
    assignments={a['image_id']:a for a in split['assignments']}
    if len(assignments) != len(split['assignments']): raise ValueError('Duplicate split assignment')
    if len(assignments)!=30 or set(assignments)!={r['image_id'] for r in s.data['images']}: raise ValueError('Incomplete split')
    grouped=defaultdict(set)
    for r in s.data['images']:
        a=assignments[r['image_id']]
        if r['partition']!=a['partition']: raise ValueError('Partition changed')
        grouped[a['group_id']].add(a['partition'])
    if any(len(v)!=1 for v in grouped.values()): raise ValueError('Group leakage')
    if stage=='develop':
        settings=yaml.safe_load(Path(config).read_text())
        candidates=development_candidates(settings)
        chosen=None
    else:
        d=Path(development)
        lock=json.loads((d/'selection.json').read_text())
        if json.loads((d/'run_status.json').read_text())['status']!='completed': raise ValueError('Development incomplete')
        if lock['reference_sha256']!=digest(s.path) or lock['code_hashes']!=code_hashes() or lock['split_sha256']!=digest(Path(references)/'split.json'): raise ValueError('Selection inputs/code changed')
        settings=lock['settings']; chosen=lock['selected']
        candidates=[chosen]+[dict(low_ratio=l,background_kernel_size=61,min_component_area=25) for l in (.5,.65)]
        candidates=[dict(t) for t in dict.fromkeys(tuple(c.items()) for c in candidates)]
    selection_metric = settings.get('selection_metric', 'f1')
    if selection_metric not in ('f1', 'mean_folder_f1'):
        raise ValueError('selection_metric must be f1 or mean_folder_f1')
    if type(settings.get('review_all_candidates', False)) is not bool:
        raise ValueError('review_all_candidates must be Boolean')
    rows=[r for r in s.data['images'] if r['partition']==('development' if stage=='develop' else 'evaluation')]
    if len(rows)!=(20 if stage=='develop' else 10): raise ValueError('Wrong partition coverage')
    out=fresh(output)
    write(out/'run_status.json',dict(status='running'))
    write(out/'metadata.json',dict(stage=stage,reference_sha256=digest(s.path),split_sha256=digest(Path(references)/'split.json'),
        grouping_method=split.get('grouping_method','inferred_filename_duplicate'),
        group_counts=split.get('group_counts'),identity_coverage=split.get('identity_coverage'),
        code_hashes=code_hashes(),settings=settings,candidates=candidates,
        python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,scipy=scipy.__version__,
        config_sha256=digest(config) if config else None))
    records=[]; predictions=[]; rankings=[]
    try:
        for ci,c in enumerate(candidates):
            print(stage,ci+1,'/',len(candidates),c,flush=True)
            for row in rows:
                result=detect_regions(read_image(s.verify_source(row)),**c,max_side=settings['max_side'],margin_ratio=settings['margin_ratio'])
                predictions.append(dict(candidate=ci,image_id=row['image_id'],**result))
                for threshold in (.5,.75):
                    scored=score_boxes(result['regions'],row,threshold)
                    records.append(dict(candidate=ci,image_id=row['image_id'],source_folder=row['source_folder'],threshold=threshold,**scored,**metrics(scored['tp'],scored['fp'],scored['fn'])))
            subset=[r for r in records if r['candidate']==ci and r['threshold']==.5]
            rankings.append(dict(candidate=ci,**c,**ranking_scores(subset)))
        winner=select_winner(rankings, selection_metric)
        if stage=='develop':
            chosen=candidates[winner['candidate']]
            write(out/'selection.json',dict(selected=chosen,selection_metric=selection_metric,settings=settings,reference_sha256=digest(s.path),split_sha256=digest(Path(references)/'split.json'),code_hashes=code_hashes()))
        write(out/'predictions.json',predictions); write(out/'scores.json',records)
        pd.DataFrame(rankings).to_csv(out/'rankings.csv',index=False)
        pd.DataFrame([{k:v for k,v in r.items() if k not in ('matches','ignored_indices')} for r in records]).to_csv(out/'per_image_scores.csv',index=False)
        summarize(out,records,rows)
        report_ids=[i for i,c in enumerate(candidates) if settings.get('review_all_candidates', False) or c==chosen or (c['background_kernel_size']==61 and c['min_component_area']==25)]
        galleries(out,s,rows,predictions,records,report_ids)
        pd.DataFrame([dict(image_id=r['image_id'],partition=r['partition'],status='processed') for r in rows]).to_csv(out/'coverage.csv',index=False)
        pd.DataFrame(columns=['image_id','error']).to_csv(out/'errors.csv',index=False)
        write(out/'verification.json',dict(images=len(rows),candidates=len(candidates),score_rows=len(records),processing_errors=0,visual_review='pending'))
        write(out/'run_status.json',dict(status='completed'))
        lines=['# Automatic detection pilot','',f'Stage: {stage}. Images: {len(rows)}. Configurations: {len(candidates)}.',f'Development-selected settings: {chosen}.','',pd.DataFrame(rankings).to_string(index=False),'',
            f'Selection uses development {selection_metric} at IoU 0.50 only. Evaluation rankings do not change selection. Mean-folder scores weight sampling folders equally, not individual species.',
            'Grouping method: '+split.get('grouping_method','inferred_filename_duplicate')+'. Parent-only identities do not establish specimen independence. Slide/session independence and microscope deployment validity remain unverified.',
            'No empty fields exist in this reference set. Unknown-organism rejection is not validated.',
            'Possible split/merge counts are overlap heuristics, not biological judgments.',
            'Manual crop review is pending. See review.md and visual_review.csv.']
        (out/'findings.md').write_text('\n\n'.join(lines),encoding='utf-8')
    except Exception as error:
        write(out/'run_status.json',dict(status='failed',error=str(error)))
        write(out/'errors.json',[dict(error=str(error))])
        (out/'selection.json').unlink(missing_ok=True)
        raise
    return out


def summarize(out,records,rows):
    summaries=[]; recalls=[]
    lookup={r['image_id']:r for r in rows}
    for ci,threshold in sorted({(r['candidate'],r['threshold']) for r in records}):
        subset=[r for r in records if r['candidate']==ci and r['threshold']==threshold]
        for folder in ['all']+sorted({r['source_folder'] for r in rows}):
            items=[r for r in subset if folder=='all' or r['source_folder']==folder]
            totals={k:sum(r[k] for r in items) for k in ('tp','fp','fn')}
            summaries.append(dict(candidate=ci,threshold=threshold,source_folder=folder,**totals,**metrics(**totals)))
        counters=defaultdict(lambda:[0,0])
        for r in subset:
            matched={m['reference'] for m in r['matches']}
            for o in lookup[r['image_id']]['objects']:
                if o['uncertain']:continue
                for category in ('species:'+o['species'],'kind:'+o['kind'],'truncated:'+str(o['truncated'])):
                    counters[category][0]+=o['object_id'] in matched; counters[category][1]+=1
        for category,(tp,total) in counters.items(): recalls.append(dict(candidate=ci,threshold=threshold,category=category,matched=tp,total=total,recall=tp/total))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(recalls).to_csv(out/'reference_recall.csv',index=False)


def galleries(out,session,rows,predictions,records,ids):
    from PIL import Image,ImageDraw
    crop_rows=[]; review=[]; links=['# Crop review','', 'Human decisions required: completeness, contamination, colony integrity and ambiguity.','']
    for index,row in enumerate(rows):
        raw=read_image(session.verify_source(row))
        for ci in ids:
            result=next(p for p in predictions if p['candidate']==ci and p['image_id']==row['image_id'])
            scored=next(r for r in records if r['candidate']==ci and r['image_id']==row['image_id'] and r['threshold']==.5)
            image=Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))
            zoom=min(4,1000/image.width,800/image.height)
            image=image.resize((max(1,round(image.width*zoom)),max(1,round(image.height*zoom))))
            sx,sy=image.width/row['width'],image.height/row['height']; draw=ImageDraw.Draw(image)
            used={m['reference'] for m in scored['matches']}
            for obj in row['objects']:
                color='gray' if obj['uncertain'] else ('lime' if obj['object_id'] in used else 'red')
                l,t,r,b=obj['box']; draw.rectangle((l*sx,t*sy,(r-1)*sx,(b-1)*sy),outline=color,width=2)
            for pi,obj in enumerate(result['regions']):
                l,t,r,b=obj['box']; draw.rectangle((l*sx,t*sy,(r-1)*sx,(b-1)*sy),outline='orange',width=1)
                crop=raw[obj['crop_box'][1]:obj['crop_box'][3],obj['crop_box'][0]:obj['crop_box'][2]]
                parent = Path(row['image_id'])
                relative = Path('crops')/parent.parent/f'{parent.name}__candidate_{ci:02d}__crop_{pi:03d}.png'
                (out/relative).parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out/relative),crop):raise OSError('Crop save failed')
                crop_rows.append(dict(image_id=row['image_id'],parent_image_id=row['image_id'],
                                      crop_image_id=relative.as_posix(),partition=row['partition'],
                                      candidate=ci,prediction=pi,path=relative.as_posix(),**obj))
            relative=Path('galleries')/f'{index:02d}_{ci}.png'; (out/relative).parent.mkdir(exist_ok=True)
            image.save(out/relative)
            links += [f'## {row["image_id"]} / configuration {ci}',f'TP {scored["tp"]}, FP {scored["fp"]}, FN {scored["fn"]}. Green=matched reference, red=missed reference, gray=uncertain, orange=prediction.',f'![Overlay]({relative.as_posix()})','']
            links += [f'![Crop {c["prediction"]}]({Path(c["path"]).as_posix()})' for c in crop_rows if c['image_id']==row['image_id'] and c['candidate']==ci]
            review.append(dict(image_id=row['image_id'],candidate=ci,status='unreviewed',notes=''))
    pd.DataFrame(crop_rows, columns=['image_id','parent_image_id','crop_image_id','partition',
                                    'candidate','prediction','path','box','crop_box','component_area','truncated']).to_csv(out/'crop_manifest.csv',index=False)
    pd.DataFrame(review).to_csv(out/'visual_review.csv',index=False)
    (out/'review.md').write_text('\n\n'.join(links),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','develop','evaluate'])
    p.add_argument('--source-manifest',type=Path);p.add_argument('--session',type=Path);p.add_argument('--references',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--development-run',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='prepare':out=prepare(a.session,a.output_dir,a.source_manifest)
    else:out=run(a.stage,a.references,a.output_dir,a.config,a.development_run)
    print('Completed:',out)


if __name__=='__main__': main()
