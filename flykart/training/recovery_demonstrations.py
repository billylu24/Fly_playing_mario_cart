"""Validate recovery teachers on training-only student states before imitation.
ROM frames remain in ignored NPZ files; public reports contain numerical results.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from . import supervised as base
from . import action_distillation as pilot

SOURCE = pilot.OUT.parent / 'distillation_replication_v1'
OUT = SOURCE.parent / 'recovery_demonstrations_v1'


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')


def frame_hash(obs):
    return hashlib.sha256(obs.tobytes()).hexdigest()


@torch.no_grad()
def baseline(m, cfg):
    task = base.BendTask(**cfg)
    try:
        obs = task.reset()
        ctx = base.Context(m)
        frames, actions, hashes = [], [], []
        while True:
            frames.append(obs.copy())
            hashes.append(frame_hash(obs))
            action = int(ctx.step(obs).argmax())
            actions.append(action)
            msg = task.step(action)
            if msg['done']:
                return dict(actions=actions, observation_hashes=hashes, **msg['episode']), frames
            obs = msg['frame']
    finally:
        task.close()


@torch.no_grad()
def branch(m, cfg, original, takeover, teacher):
    task = base.BendTask(**cfg)
    try:
        obs = task.reset()
        frames, actions, labels = [], [], []
        waypoint = 0
        takeover_hash = None
        while True:
            t = len(actions)
            # Track waypoint throughout the actual student prefix, never reset
            # it to waypoint 0 at an already advanced recovery state.
            oracle_action, waypoint = base.oracle(task, waypoint)
            if t < takeover:
                assert frame_hash(obs) == original['observation_hashes'][t]
                action = original['actions'][t]
                label = -100
            else:
                if t == takeover:
                    takeover_hash = frame_hash(obs)
                    assert takeover_hash == original['observation_hashes'][t]
                if teacher == 'waypoint':
                    action = oracle_action
                else:
                    x = torch.as_tensor(obs,device='cuda')[None,None].float()/255
                    action = int(m.head(m.b.encoder(x))[0].argmax())
                label = action
            frames.append(obs.copy())
            actions.append(action)
            labels.append(label)
            msg = task.step(action)
            if msg['done']:
                assert takeover_hash is not None
                accepted = bool(msg['episode']['target_reached'] and msg['episode']['jumps']==0)
                return dict(teacher=teacher,takeover=takeover,accepted=accepted,
                    prefix_and_takeover_match=True,takeover_observation_sha256=takeover_hash,
                    actions=actions,supervised_decisions=sum(a>=0 for a in labels),
                    **msg['episode']), frames, labels
            obs = msg['frame']
    finally:
        task.close()


def summarize(records):
    summary = {}
    for failed_only in [False,True]:
        rows = [r for r in records if not failed_only or not r['baseline_success']]
        key = 'baseline_failures' if failed_only else 'all_eligible_branches'
        summary[key] = {}
        for teacher in ['cnn','waypoint']:
            for delay in [8,32]:
                rr=[r for r in rows if r['teacher']==teacher and r['takeover']==delay]
                summary[key][f'{teacher}_{delay}']=dict(success=sum(r['accepted'] for r in rr),total=len(rr))
    return summary


def run():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    (OUT/'data').mkdir()
    manifest_path=base.EXPERIMENT/'manifest.json'
    rows=[r for r in json.loads(manifest_path.read_text()) if r['split']=='train' and r['accepted']]
    assert len(rows)==43
    sources=[Path(__file__),Path(base.__file__),manifest_path,pilot.SOURCE/'stats.pt',pilot.lp.OUT/'k2_cnn_mlp32.pt']
    sources += [SOURCE/f'seed{s}/joint_best.pt' for s in range(3)]
    save('protocol.json',dict(starts=[r['start_id'] for r in rows],seeds=[0,1,2],takeovers=[8,32],
        selection='All 43 accepted original training starts. No validation/test/previous failure-diagnosis starts used.',
        branching='For each student seed and start, replay its exact greedy prefix; after 8/32 decisions, CNN or waypoint takes over to episode end. Skip if baseline ends at/before takeover. No RAM editing or reset at takeover; retain original episode time/stall budget.',
        waypoint='Existing controller unchanged. Its waypoint state is updated at every prefix decision on actual student positions. RAM is used only by the demonstrator, not the learned policy.',
        acceptance='Target reached and jumps==0. Store full actual prefix with -100 supervision labels; store action labels only after takeover. Save both teachers outcomes even on failure.',
        dataset_rule='For each baseline-failure seed/start/takeover: prefer successful waypoint branch, otherwise successful CNN branch; reject if both fail. No successful-baseline branches enter recovery training.',
        quality_gate='At least 12 distinct start IDs with accepted baseline-failure recoveries, spanning at least 3 original prefix frames, and all 3 action classes. Gate signals coverage only, not optimal labels or independent samples.',
        next_training='Only after gate: matched old-data continuation versus recovery-data mixture. Define and lock exact loss, schedule and new evaluation starts before training.',
        sources={str(p):base.digest(p) for p in sources}))
    stats={k:v.cuda() for k,v in torch.load(pilot.SOURCE/'stats.pt',weights_only=True).items()}
    records, baselines, skipped = [], [], []
    for seed in range(3):
        m=pilot.model(stats)
        m.load_state_dict(torch.load(SOURCE/f'seed{seed}/joint_best.pt',weights_only=True))
        m.eval(); m.action=True
        for i,row in enumerate(rows):
            cfg=row['config']; start=row['start_id']
            original,_=baseline(m,cfg)
            original.update(seed=seed,config=cfg)
            baselines.append(original)
            save('baselines.json',baselines)
            for takeover in [8,32]:
                if len(original['actions'])<=takeover:
                    skipped.append(dict(seed=seed,start_id=start,takeover=takeover,reason='baseline_already_terminal'))
                    continue
                for teacher in ['cnn','waypoint']:
                    r,frames,labels=branch(m,cfg,original,takeover,teacher)
                    r.update(seed=seed,config=cfg,baseline_success=bool(original['target_reached']))
                    if r['accepted'] and not r['baseline_success']:
                        file=f'seed{seed}_{start}_t{takeover}_{teacher}.npz'
                        np.savez_compressed(OUT/'data'/file,observations=np.stack(frames),actions=np.asarray(labels,np.int64),executed_actions=np.asarray(r['actions'],np.int64))
                        r.update(filename=file,sha256=base.digest(OUT/'data'/file),action_counts=np.bincount(np.asarray(labels)[takeover:],minlength=3).tolist())
                    records.append(r)
            save('branches.json',records);save('skipped.json',skipped);save('summary.json',summarize(records))
            print('DONE',seed,i+1,'/43',start,'baseline',original['target_reached'],'branches',len(records),flush=True)
        del m
    selected=[]
    for b in baselines:
        if b['target_reached']:continue
        for takeover in [8,32]:
            rr=[r for r in records if r['seed']==b['seed'] and r['start_id']==b['start_id'] and r['takeover']==takeover and r['accepted']]
            if rr:selected.append(min(rr,key=lambda r:r['teacher']!='waypoint'))
    save('selected_manifest.json',selected)
    starts={r['start_id'] for r in selected}
    prefixes={r['config']['start_specs'][0]['prefix_frame'] for r in selected}
    counts=np.sum([r['action_counts'] for r in selected],axis=0) if selected else np.zeros(3,dtype=int)
    gate=len(starts)>=12 and len(prefixes)>=3 and bool((counts>0).all())
    save('quality_gate.json',dict(passed=gate,selected_trajectories=len(selected),distinct_starts=len(starts),prefix_frames=sorted(prefixes),action_counts=counts.tolist(),
        warning='Success-filtered correlated recoveries on training starts; labels demonstrate a successful continuation, not per-action optimality. No generalization claim.'))
    for p,h in json.loads((OUT/'protocol.json').read_text())['sources'].items():assert base.digest(Path(p))==h
    save('status.json',dict(state='completed',baseline_episodes=len(baselines),branch_episodes=len(records),skipped_branch_pairs=len(skipped),sources_unchanged=True,quality_gate_passed=gate))


if __name__=='__main__':run()
