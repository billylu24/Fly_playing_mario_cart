"""Feasibility with the SAME three actions as the pixel policy."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from training.bend_task import BendTask,STARTS
from training.task import ROOT

points=np.array([(165,70),(100,42),(64,67),(40,113),(40,300)],float)*8
p=argparse.ArgumentParser();p.add_argument('--perturb-frames',type=int,default=6)
p.add_argument('--output',default='drive_oracle.json');args=p.parse_args()
results=[]
for spec in STARTS:
    task=BendTask(spec['id'],perturb_frames=args.perturb_frames)
    try:
        first=task.reset();waypoint=0
        start_info={k:task.start_info[k] for k in ['kart1_X','kart1_Y','kart1_direction','kart1_speed','surface']}
        while True:
            info=task.env.data.lookup_all();pos=np.array([info['kart1_X'],info['kart1_Y']],float)
            if np.linalg.norm(points[waypoint]-pos)<220:waypoint=min(waypoint+1,len(points)-1)
            delta=points[waypoint]-pos
            error=(math.atan2(delta[0],-delta[1])-info['kart1_direction']*2*math.pi/256+math.pi)%(2*math.pi)-math.pi
            m=task.step(0 if abs(error)<=.12 else 2 if error>0 else 1)
            if m['done']:
                results.append({**m['episode'],'start_info':start_info,'perturb_frames':args.perturb_frames});np.testing.assert_array_equal(first,m['frame']);break
    finally:task.close()
(ROOT/'tests/bend_validation'/args.output).write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
