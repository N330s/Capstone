"""Single-episode selection/probes around native training; no optimizer or loss changes."""
from pathlib import Path
from learning.vla.runtime_io import read_json,write_json
from learning.vla.smolvla_data import CONTRACT,LEROBOT_VERSION,sha256

ROOT=Path(__file__).resolve().parents[2]


def episode_selection(receipt,name):
    if receipt['contract']!=CONTRACT or receipt['lerobot_version']!=LEROBOT_VERSION:
        raise ValueError('Unsupported export')
    train=receipt['datasets']['train']['episodes']; val=receipt['datasets']['validation']['episodes']
    names=[r['episode'] for r in train]; others=[r['episode'] for r in val]
    if len(names)!=len(set(names)) or len(others)!=len(set(others)) or set(names)&set(others):
        raise ValueError('Duplicate/leaking split')
    if name not in names: raise ValueError('Only an existing training episode may be overfit')
    index=names.index(name)
    if train[index]['split']!='train': raise ValueError('Wrong episode split')
    return index,train[index]


def anchor_state(value):
    """Only the audit strips native n_obs_steps=1; never alter training inputs."""
    import numpy as np
    array=np.asarray(value)
    if array.shape==(1,16): return array[0]
    if array.shape==(16,): return array
    raise ValueError('Unexpected native state/history shape')


class SingleEpisodeMonitor:
    def __init__(self,native,name,output,emit,*,export=None,collection=None):
        self.name,self.output,self.emit=name,Path(output),emit
        export=Path(export or ROOT/'data/drive_v2c30_20261012/export_all_v1').resolve()
        self.receipt=read_json(export/'openarm_export.json')
        self.index,self.record=episode_selection(self.receipt,name)
        self.raw=Path(collection or ROOT/'data/drive_v2c30_20261012/merged_all_v1')/name
        if sha256(self.raw/'episode.npz')!=self.record['data_sha256']: raise ValueError('Raw source changed')
        self.dataset=None; self.probes=[]; self.bound=False
        original=native.make_dataset
        def make_dataset(cfg):
            import numpy as np
            if cfg.dataset.episodes!=[self.index] or Path(cfg.dataset.root).resolve()!=export/'train':
                raise ValueError('Native episode filter/export differs')
            dataset=original(cfg)
            if dataset.num_episodes!=1 or len(dataset)!=self.record['steps']:
                raise ValueError('Native subset has wrong length')
            ids=dataset.hf_dataset['episode_index']
            if {int(v) for v in ids}!={self.index}: raise ValueError('Subset contains another episode')
            with np.load(self.raw/'episode.npz') as raw:
                states,actions,labels=raw['observation_state'],raw['action_applied'],raw['phase']
                self.probes=[('initial',0)]
                for phase in ('survey','transit_pick','descend','close','lift','insert'):
                    indices=np.flatnonzero(labels==phase)
                    if not len(indices): raise ValueError('Missing phase '+phase)
                    self.probes.append((phase,int(indices[len(indices)//2])))
                self.probes.append(('preclose',max(0,int(np.flatnonzero(labels=='close')[0])-10)))
                for phase,step in self.probes+[('terminal',len(actions)-1)]:
                    sample=dataset[step]; end=np.minimum(np.arange(step,step+50),len(actions)-1)
                    np.testing.assert_allclose(anchor_state(sample['observation.state']),states[step],atol=1e-6,rtol=0)
                    np.testing.assert_allclose(sample['action'],actions[end],atol=1e-6,rtol=0)
                    np.testing.assert_array_equal(sample['action_is_pad'],np.arange(step,step+50)>=len(actions))
            self.dataset=dataset
            self.emit({'type':'single_episode_dataset','episode':name,'episode_index':self.index,
                'frames':len(dataset),'raw_export_anchors':len(self.probes)+1,'normalization_scope':'unchanged full training split',
                'native_episode_filter':True,'full_trajectory_retained':True})
            return dataset
        native.make_dataset=make_dataset

    def bind(self,cfg,policy,processors):
        import torch
        from safetensors import safe_open
        self.cfg,self.processors=cfg,processors
        checkpoint=Path(policy.config.pretrained_path)
        checked=0; loaded=policy.state_dict()
        with safe_open(str(checkpoint/'model.safetensors'),framework='pt') as saved:
            for name in saved.keys():
                tensor=saved.get_tensor(name); actual=loaded[name].detach().cpu()
                if tensor.dtype!=actual.dtype or not torch.equal(tensor,actual):
                    raise ValueError('Initial native tensor differs: '+name)
                checked+=1
        self.bound=True
        self.emit({'type':'single_episode_initialization','episode':self.name,
            'checkpoint':str(checkpoint),'weights_sha256':sha256(checkpoint/'model.safetensors'),
            'all_saved_tensors_exact':True,'checked_tensors':checked,'optimizer_resume':False})
        self.probe(policy,0)

    def probe(self,policy,step):
        import numpy as np
        import torch
        from safetensors.torch import load_file
        from torch.utils.data import default_collate
        from learning.vla.action_metrics import errors
        if not self.bound: raise ValueError('Probe before data/model binding')
        checkpoint=Path(policy.config.pretrained_path)
        saved=load_file(str(checkpoint/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
        was_training=policy.training
        deterministic=torch.are_deterministic_algorithms_enabled()
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark=torch.backends.cudnn.benchmark; tf32=torch.backends.cuda.matmul.allow_tf32
        rows=[]
        try:
            torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
            torch.backends.cuda.matmul.allow_tf32=False; policy.eval()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for phase,index in self.probes:
                    raw=default_collate([self.dataset[index]]); batch=self.processors[0](raw)
                    for key in ('observation.state','action'):
                        mean,std=saved[key+'.mean'],saved[key+'.std']
                        expected=(raw[key].cpu()-mean)/(std+1e-8)
                        torch.testing.assert_close(batch[key].cpu(),expected,atol=1e-5,rtol=1e-5)
                    torch.testing.assert_close(self.processors[1](batch['action']).cpu(),raw['action'].cpu(),atol=1e-5,rtol=0)
                    seed=1000+index
                    noise=torch.randn((1,50,32),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                    actor={key:value for key,value in batch.items() if key not in ('action','action_is_pad','actions_id_pad')}
                    policy.reset(); prediction=self.processors[1](policy.predict_action_chunk(actor,noise=noise))[0].float().cpu().numpy()
                    if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite generated probe')
                    losses=[]
                    for t in (.1,.5,.9):
                        loss,_=policy.forward(batch,noise=noise,time=torch.tensor([t],device='cuda'))
                        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite flow probe')
                        losses.append({'time':t,'native_loss':float(loss)})
                    rows.append({'phase':phase,'frame':index,'noise_seed':seed,
                        'errors':errors(prediction,raw['action'][0].cpu().numpy()),'flow':losses})
                policy.reset()
        finally:
            torch.use_deterministic_algorithms(deterministic,warn_only=warn_only)
            torch.backends.cudnn.benchmark=benchmark; torch.backends.cuda.matmul.allow_tf32=tf32
            policy.train(was_training)
        event={'type':'single_episode_generated_probe','step':step,'episode':self.name,'rows':rows,
            'normalization_verified_against_initializer':True,'scope':'recorded training observations, not live pickup/validation'}
        self.emit(event); write_json(self.output/'single_episode_probe.json',event)

    def verify_saved_processors(self,folder,initializer):
        import torch
        from safetensors.torch import load_file
        for name in ('policy_preprocessor_step_5_normalizer_processor.safetensors',
                     'policy_postprocessor_step_0_unnormalizer_processor.safetensors'):
            old,new=load_file(str(Path(initializer)/name)),load_file(str(Path(folder)/'pretrained_model'/name))
            if old.keys()!=new.keys() or any(not torch.equal(old[k],new[k]) for k in old):
                raise ValueError('Saved normalization changed')
        write_json(Path(folder)/'single_episode_normalization.json',{'unchanged':True,'initializer':str(initializer)})
