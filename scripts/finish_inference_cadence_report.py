"""Bounded local postprocessor: wait for cadence terminal status, audit and plot.

No model, robot commands, optimizer, retries or API calls. Allows the supervised
experiment to finish and produce evidence without keeping an AI turn active.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learning.vla.runtime_io import read_json,write_json,write_status
from learning.vla.smolvla_data import sha256
from learning.vla.source_archive import archive_source
from scripts.run_smolvla_job import arm_deadline,kill_owned_process


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot-python',type=Path,required=True); args=p.parse_args()
    campaign=args.campaign.resolve(); output=args.output.resolve()
    if campaign.parent!=ROOT/'outputs' or output.parent!=ROOT/'results' or output.exists():
        raise ValueError('Existing campaign and fresh dedicated results directory required')
    status_path=campaign/'report_status.json'
    if status_path.exists(): raise ValueError('Postprocessor already requested; no automatic retry')
    plan=read_json(campaign/'plan.json'); health=read_json(campaign/'health.json')
    remaining=max(0.,plan['total_limit_s']-health['elapsed_s'])
    started=time.monotonic(); deadline=started+remaining+60
    reporter=ROOT/'scripts/report_inference_cadence.py'; reporter_hash=sha256(reporter)
    write_json(campaign/'report_request.json',{'pid':os.getpid(),'remaining_campaign_s':remaining,
        'wait_grace_s':60,'postprocessing_timeout_s':120,'output':str(output),
        'reporter':archive_source('scripts/report_inference_cadence.py'),
        'watcher':archive_source('scripts/finish_inference_cadence_report.py'),
        'scope':'Local terminal-status watcher and saved-artifact audit/plots only; no model/physics/optimizer calls.'})
    try:
        while not (campaign/'completion.json').exists() and not (campaign/'failure.json').exists():
            if time.monotonic()>deadline: raise TimeoutError('Campaign terminal-status allowance exhausted')
            write_status(status_path,{'running':True,'phase':'waiting_for_campaign',
                'elapsed_s':time.monotonic()-started,'remaining_wait_s':deadline-time.monotonic()})
            time.sleep(10)
        if sha256(reporter)!=reporter_hash: raise ValueError('Frozen reporter source changed')
        if shutil.disk_usage(ROOT).free<25*2**30: raise RuntimeError('Postprocessing deferred: disk reserve below 25 GiB')
        cmd=[str(args.plot_python.resolve()),str(reporter),'--campaign',str(campaign),'--output',str(output)]
        if (campaign/'failure.json').exists(): cmd.append('--allow-incomplete')
        write_status(status_path,{'running':True,'phase':'auditing_saved_motion_and_plotting'})
        with (campaign/'report.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            fired=threading.Event(); timer=arm_deadline(child,120,fired)
            try: code=child.wait()
            finally:
                timer.cancel()
                if child.poll() is None: kill_owned_process(child)
                child.wait()
        if code or fired.is_set(): raise RuntimeError('Postprocessing failed; see retained report.log')
        summary=read_json(output/'summary.json')
        write_json(status_path,{'running':False,'completed':True,'campaign_complete':summary['completed'],
            'output':str(output),'prefix_counts':summary['prefix_counts'],'elapsed_s':time.monotonic()-started})
        print(summary['prefix_counts'],flush=True)
    except BaseException as exc:
        write_json(status_path,{'running':False,'completed':False,'error':repr(exc),
            'elapsed_s':time.monotonic()-started,'no_automatic_retry':True})
        raise


if __name__=='__main__': main()
