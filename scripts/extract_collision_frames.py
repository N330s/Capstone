"""Extract unchanged frames from the original saved rollout videos; never reruns a policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import write_json
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--diagnosis',type=Path,required=True)
    args=p.parse_args(); out=args.diagnosis/'frames'; out.mkdir(exist_ok=False)
    report=json.loads((args.diagnosis/'report.json').read_text()); rows=[]
    import av
    for name in ('train_20003','train_22045','train_20042'):
        row=next(r for r in report['rows'] if r['episode']==name); events=row['chronology']
        targets={'first_contact':events['first_one_or_two_contacts'],
                 'middle':events['maximum_lift_command'] if not row['pickup_success'] else events['first_grasp'],
                 'terminal':row['steps']}
        video=ROOT/'outputs/five_episode_openhold_20261020'/('training_policy_'+name)/'rollout.mp4'
        saved=[]
        with av.open(str(video)) as container:
            for i,frame in enumerate(container.decode(video=0)):
                for label,index in targets.items():
                    if i==index:
                        path=out/f'{name}_{label}_{i:04d}.png'; frame.to_image().save(path)
                        saved.append({'label':label,'command':i,'image':str(path.resolve())})
                if i>=max(targets.values()): break
        if len(saved)!=len(targets): raise ValueError('Missing original video frame')
        rows.append({'episode':name,'video_sha256':sha256(video),'frames':saved})
    write_json(args.diagnosis/'frames_manifest.json',{'rows':rows,'implementation':archive_source('scripts/extract_collision_frames.py'),
        'limits':'Unchanged frames decoded from original video, not reconstructed motion or new execution. Display headers omit inference pauses.'})


if __name__=='__main__': main()
