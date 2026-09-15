"""Privileged waypoint controller solely for validating race semantics."""
import os
os.environ.pop('DISPLAY',None)
from pathlib import Path
import json,math
import cv2
import numpy as np
import stable_retro as retro
root=Path(__file__).resolve().parents[1]
out=root/'tests'/'track_validation';out.mkdir(exist_ok=True)
retro.data.Integrations.add_custom_path(str(root/'mario_kart'))
env=retro.make('SuperMarioKart-Snes',inttype=retro.data.Integrations.CUSTOM_ONLY,
               scenario=str(root/'mario_kart/validation_scenario.json'),render_mode='rgb_array')
points=np.array([(464,260),(454,224),(420,206),(350,170),(260,120),(165,70),(100,42),(64,67),
                 (40,113),(40,300),(42,340),(65,366),(100,360),(200,315),(245,302),(265,318),
                 (312,400),(345,438),(385,452),(423,440),(456,415),(466,370),(464,310)],dtype=float)*8
try:
    env.reset();_,_,_,_,info=env.step(np.zeros(12,dtype=np.int8))
    waypoint=0;trace=[];events=[];prev=None
    for frame in range(18000):
        pos=np.array([info['kart1_X'],info['kart1_Y']],dtype=float)
        target=points[waypoint]
        if np.linalg.norm(target-pos)<220:
            waypoint=(waypoint+1)%len(points);target=points[waypoint]
        desired=math.atan2(target[0]-pos[0],-(target[1]-pos[1]))
        heading=info['kart1_direction']*2*math.pi/256
        error=(desired-heading+math.pi)%(2*math.pi)-math.pi
        action=np.zeros(12,dtype=np.int8)
        action[env.buttons.index('Y' if info['kart1_speed']>450 else 'B')]=1
        if abs(error)>0.12:action[env.buttons.index('RIGHT' if error>0 else 'LEFT')]=1
        if frame%4==0:held_action=action.copy()
        obs,_,_,_,info=env.step(held_action)
        trace.append({'frame':frame,'action':held_action.tolist(),**info})
        key=(info['lap'],info['current_checkpoint'],info['getGameMode'])
        if key!=prev:
            print(frame,'lap/cp/mode',key,'pos',info['kart1_X'],info['kart1_Y'],'heading',info['kart1_direction'],'wp',waypoint,flush=True)
            events.append({'frame':frame,**info});prev=key
        if frame%600==0:cv2.imwrite(str(out/'controller_latest.png'),cv2.cvtColor(obs,cv2.COLOR_RGB2BGR))
        if info['lap']>=133:
            cv2.imwrite(str(out/'finish.png'),cv2.cvtColor(obs,cv2.COLOR_RGB2BGR));break
    (out/'race_trace.json').write_text(json.dumps({'events':events,'samples':trace,'last':info},indent=2)+'\n')
    if info['lap']>=133:
        for _ in range(240):obs,_,_,_,_=env.step(np.zeros(12,dtype=np.int8))
        cv2.imwrite(str(out/'finish_confirmation.png'),cv2.cvtColor(obs,cv2.COLOR_RGB2BGR))
    print('FINAL',info,flush=True)
finally:env.close()
