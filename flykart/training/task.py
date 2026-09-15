"""Finite-horizon Time Trial objective, independent from upstream Lua rewards."""
from pathlib import Path
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ACTIONS=[[],['B'],['B','LEFT'],['B','RIGHT'],['Y'],['Y','LEFT'],['Y','RIGHT']]


class Progress:
    def __init__(self,limit=18000,target=150,stall_limit=0,time_penalty=.00025):
        if not 1<=target<=150 or limit<1 or stall_limit<0:raise ValueError('invalid task limits')
        self.target=target;self.stall_limit=stall_limit;self.last_advance=0
        if not 0<=time_penalty<1:raise ValueError('invalid time penalty')
        self.time_penalty=time_penalty
        self.limit=limit;self.frames=0;self.started=False;self.frontier=0
        self.previous=(-1,29);self.done=False;self.jumps=0;self.last_q=None
    def update(self,info):
        if self.done:raise RuntimeError('step after terminal')
        self.frames+=1;reward=-self.time_penalty
        lap=int(info['lap'])-128;cp=int(info['current_checkpoint'])
        if int(info['lapsize'])!=30 or not (-1<=lap<=5 and 0<=cp<30):
            raise RuntimeError(f'invalid lap/checkpoint: {lap}/{cp}')
        if not self.started and self.previous==(-1,29) and (lap,cp)==(0,0):
            self.started=True
        q=lap*30+cp
        if self.started and 0<=q<=150:
            if q==self.frontier+1 and self.last_q==self.frontier:
                self.frontier=q;reward+=1/30;self.last_advance=self.frames
            elif q>self.frontier+1 and q!=self.last_q:self.jumps+=1
        self.last_q=q;self.previous=(lap,cp)
        reason=None
        if self.frontier>=self.target and self.target<150:
            reward+=5*self.target/150;reason='target_reached'
        elif lap>=5:
            if self.frontier!=150:raise RuntimeError('finish without sequential validated progress')
            reward+=5;reason='success'
        elif int(info['getGameMode'])!=28:reason='left_race'
        elif self.stall_limit and self.frames-self.last_advance>=self.stall_limit:reason='no_progress'
        elif self.frames>=self.limit:reason='timeout'
        self.done=reason is not None
        return reward,reason


class Task:
    def __init__(self,limit=18000,target=150,stall_limit=0,repeat=4,action_set='full',time_penalty=.00025):
        if repeat<1 or action_set not in ('full','drive'):raise ValueError('invalid actions')
        self.target=target;self.stall_limit=stall_limit;self.repeat=repeat
        self.time_penalty=time_penalty
        os.environ.pop('DISPLAY',None);os.environ.pop('WAYLAND_DISPLAY',None)
        os.environ['SDL_AUDIODRIVER']='dummy'
        import stable_retro as retro
        import cv2
        cv2.setNumThreads(1)
        retro.data.Integrations.add_custom_path(str(ROOT/'mario_kart'))
        self.env=retro.make('SuperMarioKart-Snes',inttype=retro.data.Integrations.CUSTOM_ONLY,
                            scenario=str(ROOT/'mario_kart/validation_scenario.json'),render_mode='rgb_array')
        self.limit=limit;self.vectors=[]
        for names in (ACTIONS if action_set=='full' else ACTIONS[1:4]):
            a=np.zeros(12,dtype=np.int8)
            for name in names:a[self.env.buttons.index(name)]=1
            self.vectors.append(a)
    @staticmethod
    def frame(obs):
        import cv2
        return cv2.resize(cv2.cvtColor(obs,cv2.COLOR_RGB2GRAY),(84,84),interpolation=cv2.INTER_AREA)
    def reset(self):
        obs,_=self.env.reset();self.progress=Progress(self.limit,self.target,self.stall_limit,self.time_penalty);self.total_reward=0.
        return self.frame(obs)
    def step(self,action):
        reward=0.;reason=None
        for count in range(1,self.repeat+1):
            obs,_,_,_,info=self.env.step(self.vectors[action])
            r,reason=self.progress.update(info);reward+=r
            if reason:break
        self.total_reward+=reward
        episode=None
        if reason:
            episode={'reason':reason,'frames':self.progress.frames,'progress':self.progress.frontier,
                     'one_lap':self.progress.frontier>=30,'reward':self.total_reward,'jumps':self.progress.jumps,
                     'target_reached':self.progress.frontier>=self.target,'target':self.target}
            frame=self.reset()
        else:frame=self.frame(obs)
        return {'frame':frame,'reward':reward,'done':bool(reason),'frames':count,'episode':episode,
                'progress':self.progress.frontier,'info':{k:info[k] for k in ['lap','current_checkpoint','kart1_speed','surface']}}
    def close(self):self.env.close()


def make_task(task_config=None):
    cfg=dict(task_config or {})
    if cfg.pop('kind','race')=='bend':
        from .bend_task import BendTask
        return BendTask(**cfg)
    return Task(**cfg)


def worker(pipe,task_config=None):
    task=None
    try:
        task=make_task(task_config);pipe.send({'frame':task.reset()})
        while True:
            cmd=pipe.recv()
            if cmd=='close':break
            if cmd=='reset':pipe.send({'frame':task.reset()})
            else:pipe.send(task.step(int(cmd)))
    except Exception as exc:
        try:pipe.send({'error':repr(exc)})
        except (BrokenPipeError,EOFError):pass
    finally:
        if task:task.close()
        pipe.close()
