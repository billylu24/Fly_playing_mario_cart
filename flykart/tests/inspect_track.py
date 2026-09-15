"""Read the existing integration RAM map for task validation, not policy input."""
from pathlib import Path
import os
os.environ.pop('DISPLAY',None)
import numpy as np
import cv2
import stable_retro as retro
root=Path(__file__).resolve().parents[1]
retro.data.Integrations.add_custom_path(str(root/'mario_kart'))
env=retro.make('SuperMarioKart-Snes',inttype=retro.data.Integrations.CUSTOM_ONLY,
               scenario=str(root/'mario_kart/validation_scenario.json'),render_mode='rgb_array')
try:
    env.reset();_,_,_,_,info=env.step(np.zeros(12,dtype=np.int8))
    print('blocks',[(k,len(v)) for k,v in env.data.memory.blocks.items()])
    ram=np.frombuffer(env.data.memory.blocks[0x7e0000],dtype=np.uint8)
    print('info',info,'RAM',ram.shape)
    tiles=ram[0x10000:0x14000].reshape(128,128)
    physics=ram[0xb00:0xc00][tiles]
    out=root/'tests'/'track_validation';out.mkdir(exist_ok=True)
    np.savez(out/'map.npz',tiles=tiles,physics=physics)
    cv2.imwrite(str(out/'tiles.png'),cv2.resize(tiles*17,(512,512),interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(str(out/'physics.png'),cv2.resize(physics,(512,512),interpolation=cv2.INTER_NEAREST))
    print('physics codes',np.unique(physics,return_counts=True))
finally:env.close()
