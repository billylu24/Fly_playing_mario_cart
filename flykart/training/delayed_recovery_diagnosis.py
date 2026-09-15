"""Can the frozen CNN recover from states reached by a failing graph policy?"""
import json
from pathlib import Path
import torch
from . import divergence_diagnosis as prior
from . import supervised as base

OUT = prior.OUT.parent / 'delayed_recovery_diagnosis_v1'


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2)+'\n')


@torch.no_grad()
def rollout(m, cfg, baseline, takeover):
    task = base.BendTask(**cfg)
    try:
        obs = task.reset()
        ctx = base.Context(m)
        actions, steps = [], []
        while True:
            t = len(actions)
            recon = ctx.step(obs)
            z = m.b.encoder(torch.as_tensor(obs,device='cuda')[None,None].float()/255)[0]
            sp = m.head(recon[None]*m.zscale+m.zmean)[0].softmax(-1)
            tp = m.head(z[None])[0].softmax(-1)
            student, teacher = int(sp.argmax()), int(tp.argmax())
            action = student if t < takeover else teacher
            if t < takeover:
                assert action == baseline['actions'][t], (t,action,baseline['actions'][t])
            msg = task.step(action)
            steps.append(dict(t=t,action=action,student_action=student,teacher_action=teacher,
                student_prob=sp.cpu().tolist(),teacher_prob=tp.cpu().tolist(),after=msg['info']))
            actions.append(action)
            if msg['done']:
                assert len(actions)>takeover
                return dict(takeover=takeover,prefix_matches_baseline=True,actions=actions,steps=steps,**msg['episode'])
            obs = msg['frame']
    finally:
        task.close()


def run():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    cases = json.loads((prior.OUT/'summary.json').read_text())
    manifest = json.loads((prior.SOURCE/'new_test_manifest.json').read_text())
    configs = {r['config']['start_id']:r['config'] for r in manifest}
    sources = [Path(__file__),Path(prior.__file__),prior.OUT/'summary.json',prior.pilot.SOURCE/'stats.pt']
    sources += [prior.SOURCE/f'seed{s}/joint_best.pt' for s in range(3)]
    save('protocol.json',dict(delays=[8,32],
        intervention='Student until first disagreement + 8 or +32 decisions, then frozen direct CNN to end. Same actual observation history, no RAM input or state edits.',
        scope='Follow-up after first-disagreement study. Same two selected failure cases; diagnostic teacher-assisted control, not autonomous graph success.',
        sources={str(p):base.digest(p) for p in sources}))
    stats = {k:v.cuda() for k,v in torch.load(prior.pilot.SOURCE/'stats.pt',weights_only=True).items()}
    summary=[]
    for seed in range(3):
        m=prior.pilot.model(stats)
        m.load_state_dict(torch.load(prior.SOURCE/f'seed{seed}/joint_best.pt',weights_only=True))
        m.eval()
        for case in [c for c in cases if c['seed']==seed]:
            start=case['start']
            baseline=json.loads((prior.OUT/f'seed{seed}_{start}_student.json').read_text())
            for delay in [8,32]:
                takeover=case['first_disagreement']+delay
                r=rollout(m,configs[start],baseline,takeover)
                save(f'seed{seed}_{start}_delay{delay}.json',r)
                summary.append(dict(seed=seed,start=start,delay=delay,**{k:r[k] for k in ['takeover','target_reached','reason','frames','progress']}))
                save('summary.json',summary)
                print('DONE',seed,start,delay,r['target_reached'],r['progress'],flush=True)
        del m
    for p,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():
        assert base.digest(Path(p))==h
    save('status.json',dict(state='completed',episodes=12,prefix_checks_passed=True,sources_unchanged=True))


if __name__=='__main__':
    run()
