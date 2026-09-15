"""Paired trajectory shadows and prespecified first-disagreement interventions.
No training, RAM policy input, state editing, or saved game images.
"""
import json
from pathlib import Path
import torch
from . import supervised as base
from . import action_distillation as pilot

SOURCE = pilot.OUT.parent / 'distillation_replication_v1'
OUT = SOURCE.parent / 'divergence_diagnosis_v1'
STARTS = ['f455_left_p38', 'f505_left_p14']


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n')


def archived(name, start):
    episodes = json.loads((SOURCE / (name + '_evaluation.json')).read_text())['episodes']
    return next(e for e in episodes if e['start_id'] == start and e['mode'] == 'normal' and e['sample_seed'] is None)


@torch.no_grad()
def rollout(m, cfg, driver, intervention=None):
    task = base.BendTask(**cfg)
    try:
        obs = task.reset()
        ctx = base.Context(m)
        steps, actions = [], []
        first = None
        while True:
            # Use deployment's raw-image Context, including its 64-frame rebuild.
            recon = ctx.step(obs)
            z = m.b.encoder(torch.as_tensor(obs, device='cuda')[None, None].float() / 255)[0]
            sl = m.head(recon[None] * m.zscale + m.zmean)[0]
            tl = m.head(z[None])[0]
            sp, tp = sl.softmax(-1), tl.softmax(-1)
            sa, ta = int(sp.argmax()), int(tp.argmax())
            t = len(actions)
            if sa != ta and first is None:
                first = t
            use_teacher = driver == 'teacher' or (intervention is not None and first is not None and t < first + intervention)
            action = ta if use_teacher else sa
            record = dict(t=t, student_action=sa, teacher_action=ta, action=action,
                          student_prob=sp.cpu().tolist(), teacher_prob=tp.cpu().tolist(),
                          teacher_used=use_teacher,
                          normalized_mse=float((recon - (z-m.zmean)/m.zscale).square().mean()),
                          teacher_kl=float((tp * (tl.log_softmax(-1)-sl.log_softmax(-1))).sum()))
            msg = task.step(action)
            record['after'] = msg['info']
            steps.append(record)
            actions.append(action)
            if msg['done']:
                return dict(driver=driver, intervention=intervention, first_disagreement=first,
                            actions=actions, steps=steps, **msg['episode'])
            obs = msg['frame']
    finally:
        task.close()


def run():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    paths = [Path(__file__), pilot.SOURCE/'stats.pt', pilot.lp.OUT/'k2_cnn_mlp32.pt']
    paths += [SOURCE/f'seed{s}/joint_best.pt' for s in range(3)]
    save('protocol.json', dict(starts=STARTS, seeds=[0,1,2],
         selection='Two reused diagnostic starts where all joint seeds fail and CNN succeeds; not an unbiased success estimate.',
         interventions='At first student/teacher greedy disagreement: teacher for 1 decision, 4 decisions, or remaining episode (225 upper bound). Then student resumes; all observations and histories follow actual executed actions.',
         controls='Unmodified student and teacher trajectories; exact archived action/reason/frames/progress/jumps regression for all six pairs. Teacher is frozen direct CNN head.',
         steps='One decision repeats at most 8 game frames. No training or checkpoint selection.',
         sources={str(p):base.digest(p) for p in paths}))
    manifest = json.loads((SOURCE/'new_test_manifest.json').read_text())
    configs = {r['config']['start_id']:r['config'] for r in manifest}
    stats = {k:v.cuda() for k,v in torch.load(pilot.SOURCE/'stats.pt',weights_only=True).items()}
    summary = []
    for seed in range(3):
        m = pilot.model(stats)
        m.load_state_dict(torch.load(SOURCE/f'seed{seed}/joint_best.pt',weights_only=True))
        m.eval()
        for start in STARTS:
            results = {}
            for driver, length in [('student',None),('teacher',None),('student',1),('student',4),('student',225)]:
                name = driver if length is None else f'correct_{length}'
                r = rollout(m, configs[start], driver, length)
                if length is None:
                    ref = archived('cnn' if driver=='teacher' else f'joint_seed{seed}',start)
                    for key in ['actions','reason','frames','progress','jumps']:
                        assert r[key] == ref[key], (seed,start,driver,key)
                    r['archive_regression_passed'] = True
                results[name] = r
                save(f'seed{seed}_{start}_{name}.json',r)
                print('DONE', seed, start, name, r['target_reached'], r['first_disagreement'], flush=True)
            d = results['student']['first_disagreement']
            assert d is not None
            for name in ['correct_1','correct_4','correct_225']:
                assert results[name]['actions'][:d] == results['student']['actions'][:d]
                assert results[name]['first_disagreement'] == d
            # Before first disagreement teacher and student inhabit the same states;
            # switching permanently must exactly recover the archived teacher trace.
            assert results['correct_225']['actions'] == results['teacher']['actions']
            summary.append(dict(seed=seed,start=start,first_disagreement=d,
                first_step=results['student']['steps'][d],
                outcomes={k:{x:r[x] for x in ['target_reached','frames','progress','reason']} for k,r in results.items()},
                trajectory_metrics={k:dict(decisions=len(r['steps']),
                    disagreement_rate=sum(s['student_action']!=s['teacher_action'] for s in r['steps'])/len(r['steps']),
                    mean_mse=sum(s['normalized_mse'] for s in r['steps'])/len(r['steps']),
                    mean_kl=sum(s['teacher_kl'] for s in r['steps'])/len(r['steps'])) for k,r in results.items()}))
            save('summary.json',summary)
        del m
    for p,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():
        assert base.digest(Path(p)) == h
    save('status.json',dict(state='completed',episodes=30,archive_regressions=12,
        branch_prefix_checks=True,permanent_switch_matches_teacher=True,sources_unchanged=True))


if __name__ == '__main__':
    run()
