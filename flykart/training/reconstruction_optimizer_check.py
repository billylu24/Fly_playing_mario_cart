"""Learning-rate sensitivity on identical frozen motor features, no graph updates."""
import json
import torch
from torch import nn
from . import layer_probes as lp
from .transport_reconstruction import OUT,raw_states,save

def run():
    torch.set_num_threads(3)
    save('optimizer_control_protocol.json',dict(trigger='Initial Adam lr .003 decoder update increased reconstruction MSE from .986 to34.90; isolate step-size before interpreting trainability.',rates=[.003,.0003,.00003],updates=400,rule='same zero decoder initialization, frozen motor features, normalization, optimizer eps1e-5 and clip1; report all rates, do not select by validation',purpose='attribute finite-budget fitting difficulty, not claim to fix graph or driving'))
    b,_=lp.backbone('k2');data=torch.load(OUT/'features.pt',weights_only=False);z=data['z'].cuda();v=data['validation_z'].cuda();s={k:t.cuda() for k,t in data['stats'].items()}
    with torch.no_grad():
        x=((raw_states(b,z)-s['hmean'])/s['hscale']).flatten(0,1);vx=((raw_states(b,v)-s['hmean'])/s['hscale']).flatten(0,1);y=((z-s['zmean'])/s['zscale']).flatten(0,1);vy=((v-s['zmean'])/s['zscale']).flatten(0,1)
        singular=torch.linalg.svdvals(x.double());gram=x.double()@x.double().T;torch.save(dict(x=x.cpu(),vx=vx.cpu(),y=y.cpu(),vy=vy.cpu()),OUT/'frozen_motor_features.pt')
    results={}
    for lr in [.003,.0003,.00003]:
        decoder=nn.Linear(x.shape[1],64,device='cuda');nn.init.zeros_(decoder.weight);nn.init.zeros_(decoder.bias);opt=torch.optim.Adam(decoder.parameters(),lr=lr,eps=1e-5);history=[]
        for step in range(401):
            if step in [0,1,25,100,200,400]:
                with torch.no_grad():history.append(dict(step=step,train_mse=float((decoder(x)-y).square().mean()),validation_mse=float((decoder(vx)-vy).square().mean())))
            if step==400:break
            opt.zero_grad();loss=(decoder(x)-y).square().mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.);opt.step()
        results[str(lr)]=history;torch.save(decoder.state_dict(),OUT/f'optimizer_lr{lr}.pt')
    save('optimizer_controls.json',dict(results=results,singular_values=singular.cpu().tolist(),centered_feature_rank_tolerance1e_6=int((singular>singular[0]*1e-6).sum()),initial_gradient_hessian_scale=float(torch.linalg.eigvalsh(gram).max()/len(x)),caveat='Changing learning rate changes optimization, not function class. Ridge interpolant may overfit; tiny train loss is not generalization.'))
    print(json.dumps({k:v[-1] for k,v in results.items()}),flush=True)
if __name__=='__main__':run()
