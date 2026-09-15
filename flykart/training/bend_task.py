"""Deterministic, action-replayed bend starts. No RAM edits or policy RAM input."""
import json
import numpy as np
from .task import ROOT,Task,Progress

STARTS=[{'id':f'f{frame}_{turn}','prefix_frame':frame,'turn':turn,
         'split':'heldout' if frame==570 else 'train'}
        for frame in (540,570,600) for turn in ('center','left','right')]


class BendTask(Task):
    def __init__(self,start_id='f540_center',perturb_frames=6,start_ids=None,seed=0,start_specs=None,**kwargs):
        self.specs=STARTS if start_specs is None else start_specs
        if not self.specs or len({x['id'] for x in self.specs})!=len(self.specs):
            raise ValueError('start specs must have unique IDs')
        for spec in self.specs:
            if spec['turn'] not in ('center','left','right') or not isinstance(spec['prefix_frame'],int) or spec['prefix_frame']<0:
                raise ValueError('invalid start spec')
        self.start_ids=start_ids or [start_id];self.rng=np.random.default_rng(seed)
        if not set(self.start_ids)<=set(x['id'] for x in self.specs):raise ValueError('unknown start ID')
        if perturb_frames<1:raise ValueError('invalid perturbation duration')
        self.perturb_frames=perturb_frames
        self.spec=next(x for x in self.specs if x['id']==self.start_ids[0])
        self.prefix=json.loads((ROOT/'tests/track_validation/race_trace.json').read_text())['samples']
        if any(x['prefix_frame']>=len(self.prefix) for x in self.specs):raise ValueError('prefix frame outside trace')
        super().__init__(limit=1800,target=13,stall_limit=600,repeat=8,
                         action_set=kwargs.pop('action_set','drive'),time_penalty=0,**kwargs)

    def reset(self):
        chosen=self.start_ids[int(self.rng.integers(len(self.start_ids)))]
        self.spec=next(x for x in self.specs if x['id']==chosen)
        self.env.reset()
        self.env.step(np.zeros(12,dtype=np.int8))
        for sample in self.prefix[:self.spec['prefix_frame']+1]:
            obs,_,_,_,info=self.env.step(np.asarray(sample['action'],dtype=np.int8))
        expected=self.prefix[self.spec['prefix_frame']]
        for key in ('lap','current_checkpoint','kart1_X','kart1_Y','kart1_direction'):
            if info[key]!=expected[key]:raise RuntimeError(f'prefix replay mismatch: {key}')
        a=np.zeros(12,dtype=np.int8);a[self.env.buttons.index('B')]=1
        if self.spec['turn']!='center':a[self.env.buttons.index(self.spec['turn'].upper())]=1
        for _ in range(self.perturb_frames):obs,_,_,_,info=self.env.step(a)
        q=(int(info['lap'])-128)*30+int(info['current_checkpoint'])
        if not 0<=q<self.target or info['getGameMode']!=28:raise RuntimeError('invalid bend start')
        self.start_info=dict(info);self.start_q=q
        self.progress=Progress(self.limit,self.target,self.stall_limit,self.time_penalty)
        self.progress.started=True;self.progress.frontier=q;self.progress.last_q=q
        self.progress.previous=(int(info['lap'])-128,int(info['current_checkpoint']))
        self.total_reward=0.
        return self.frame(obs)

    def step(self,action):
        start_q=self.start_q;start_id=self.spec['id']
        m=super().step(action)
        if m['done']:
            m['episode'].update(start_id=start_id,start_progress=start_q,
                                advance=m['episode']['progress']-start_q)
        return m
