#!/usr/bin/env python3
"""Depth-graded student pilot: narrowed early blocks, full late, depth-gated.

RRR-fit rank-r maps (dense reconstruct) for layers 0-5 from streamed
covariances (10 fit reals); configs A/B/C/D; full pipeline vs
HF-oracle refs on 12 scenes (6 fixture + 6 real). Pre-registered
2026-09-22: near-miss, none >=0.999.
Run: python adapt/student_probe.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import copy

LAYERS = (0, 1, 2, 3, 4, 5)
N_FIT = int(__import__('os').environ.get('STUDENT_NFIT', '10'))
LAMBDA = 1e-6
MAPS = (("q", "H", "Q", 384, 384), ("k", "H", "K", 384, 384),
        ("v", "H", "V", 384, 384), ("proj", "C", "Oattn", 384, 384),
        ("mlp1", "N2", "P1", 384, 1536), ("mlp2", "G", "Y", 1536, 384))
CONFIGS = {
    "A": {0: 96, 1: 96, 2: 96},
    "B": {0: 96, 1: 96, 2: 96, 3: 96},
    "C": {0: 192, 1: 192, 2: 192, 3: 192, 4: 192, 5: 192},
    "D": {0: 96, 1: 96, 2: 96, 3: 192, 4: 192, 5: 192},
    "E": {0: 96},
    "F": {0: 192},
}
SEQ_CONFIG = {0: 96, 1: 96, 2: 96}  # sequential chain (A ranks)


def corr(a, b):
    a = np.asanyarray(a, dtype=np.float64).flatten()
    b = np.asanyarray(b, dtype=np.float64).flatten()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return -1.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else -1.0


def rrr_from_covs(Sxx, Sxy, r):
    d1 = Sxx.shape[0]
    lam = LAMBDA * float(np.trace(Sxx)) / d1
    W_ols = np.linalg.solve(Sxx + lam * np.eye(d1), Sxy)
    M = W_ols.T @ Sxx @ W_ols
    vals, vecs = np.linalg.eigh(M)
    Vr = vecs[:, -r:] if r < M.shape[0] else vecs
    return W_ols @ Vr @ Vr.T


# torch module name per map key in backbone buffers
BUF = {"q": "q.weight", "k": "k.weight", "v": "v.weight",
       "proj": "proj.weight", "mlp1": "mlp1.weight", "mlp2": "mlp2.weight"}


def main():
    import torch
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr as _corr, SEED_CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']

    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    fit_rgb = [zr['rgb'][i].astype(np.float32) / 255.0 for i in range(N_FIT)]
    fx = load_fixtures(include_audit=False)
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate')
               for rgb, ref, cid in fx[role][:3]] +
              [(zr['rgb'][i].astype(np.float32) / 255.0,
                zr['ref'][i].astype(np.float64), f"eval-{i}")
               for i in range(N_FIT, N_FIT + 6)])
    print(f"{len(scenes)} eval scenes", flush=True)

    covs = {li: {} for li in LAYERS}
    with torch.no_grad():
        for rgb in fit_rgb:
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            for li in LAYERS:
                B = {k: v.squeeze(0).double().cpu().numpy()
                     for k, v in cap[(li, 'blk')].items()}
                for name, ik, tk, di, do in MAPS:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    Y = B[tk]
                    S = covs[li].setdefault(name, {
                        'Sxx': np.zeros((di + 1, di + 1)),
                        'Sxy': np.zeros((di + 1, do))})
                    S['Sxx'] += X.T @ X
                    S['Sxy'] += X.T @ Y
    print("covariances done.", flush=True)

    # RRR maps per config (dense reconstruct, float32)
    orig = {k: v.clone() for k, v in bb._w.items()}
    NB = sum(v.numel() for v in orig.values())
    cfgs = {"seed": {}}
    cfgs.update(CONFIGS)
    for tag, spec in cfgs.items():
        nbytes = 0
        for li, r in spec.items():
            for name, _, _, di, do in MAPS:
                S = covs[li][name]
                Wrr = rrr_from_covs(S['Sxx'], S['Sxy'], r)
                Wmat, bvec = Wrr[:-1].astype(np.float32), Wrr[-1].astype(np.float32)
                bb._w[f'layer{li}.{BUF[name]}'].copy_(
                    torch.from_numpy(Wmat.T.reshape(
                        orig[f'layer{li}.{BUF[name]}'].shape)).to(device))
                # NOTE: buffers store weight as (out,in); RRR fits (in->out)
                # rows: transpose back. Bias folded into map: teacher bias
                # REPLACED by fitted bvec — need bias buffers patched too.
                nbytes += r * (di + do)
        # patch biases from fitted intercepts
        for li, r in spec.items():
            for name, _, _, di, do in MAPS:
                S = covs[li][name]
                Wrr = rrr_from_covs(S['Sxx'], S['Sxy'], r)
                bvec = Wrr[-1].astype(np.float32)
                bkey = f'layer{li}.{BUF[name].replace(".weight", ".bias")}'
                if bkey in bb._w:
                    bb._w[bkey].copy_(torch.from_numpy(bvec).to(device))
        cs = []
        detail = []
        for rgb, ref, cid in scenes:
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            pred = run_pipeline(shared, cfg, [rgb])[0]
            c = _corr(pred, ref)
            cs.append(c)
            detail.append(f"{cid}={c:.4f}")
        full_mb = NB * 4 / 1e6
        saved_mb = sum(
            sum((di * do - r * (di + do)) * 4 for _, _, _, di, do in MAPS)
            for li, r in spec.items()) / 1e6 if spec else 0.0
        print(f"[{tag}] mean={np.mean(cs):.5f} min={min(cs):.5f} "
              f"pass={sum(c >= 0.999 for c in cs)}/{len(cs)} "
              f"~{full_mb - saved_mb:.1f}MB vs {full_mb:.1f}MB", flush=True)
        if tag in ("A", "E", "F"):
            print(f"  detail: {' '.join(detail)}", flush=True)
        for k, v in orig.items():
            bb._w[k].copy_(v)

    # ---- sequential RRR: fit each layer on STUDENT (perturbed) inputs ----
    import torch.nn.functional as _Fn
    Wfull = {k: v.double().cpu().numpy() for k, v in orig.items()}

    def _ln(X, w, b):
        mu = X.mean(axis=1, keepdims=True)
        va = ((X - mu) ** 2).mean(axis=1, keepdims=True)
        return (X - mu) / np.sqrt(va + 1e-6) * w + b

    def _gelu(X):
        import torch as _t
        with _t.no_grad():
            return _t.nn.functional.gelu(_t.from_numpy(X)).numpy()

    def _lin(X, W_):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ W_

    def _student_layer_out(Xin, Wm, li):
        n1w, n1b = Wfull[f'layer{li}.norm1.weight'], Wfull[f'layer{li}.norm1.bias']
        n2w, n2b = Wfull[f'layer{li}.norm2.weight'], Wfull[f'layer{li}.norm2.bias']
        ls1, ls2 = Wfull[f'layer{li}.ls1'], Wfull[f'layer{li}.ls2']
        H = _ln(Xin, n1w, n1b)
        Q, K, V = _lin(H, Wm['q']), _lin(H, Wm['k']), _lin(H, Wm['v'])
        outs = []
        for h in range(6):
            q, k, v = (a[:, h * 64:(h + 1) * 64] for a in (Q, K, V))
            sc = q @ k.T / 8.0
            sc = np.exp(sc - sc.max(axis=1, keepdims=True))
            outs.append(sc / sc.sum(axis=1, keepdims=True) @ v)
        Oa = _lin(np.concatenate(outs, axis=1), Wm['proj'])
        Mp = Xin + Oa * ls1
        Yp = _lin(_gelu(_lin(_ln(Mp, n2w, n2b), Wm['mlp1'])), Wm['mlp2'])
        return Mp + Yp * ls2

    # teacher block tensors per fit image (inputs AND targets)
    teach = []
    with torch.no_grad():
        for rgb in fit_rgb:
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            teach.append({li: {k: v.squeeze(0).double().cpu().numpy()
                               for k, v in cap[(li, 'blk')].items()}
                          for li in SEQ_CONFIG})
    # Xin per image = layer0 input
    with torch.no_grad():
        Xins = []
        for rgb in fit_rgb:
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            Xins.append(cap[0][0].squeeze(0).double().cpu().numpy())
    seq_W = {}
    drift = {}
    cur_in = list(Xins)  # student inputs per fit image (L0 gets teacher in)
    for li, r in SEQ_CONFIG.items():
        n1w, n1b = (Wfull[f'layer{li}.norm1.weight'],
                    Wfull[f'layer{li}.norm1.bias'])
        n2w, n2b = (Wfull[f'layer{li}.norm2.weight'],
                    Wfull[f'layer{li}.norm2.bias'])
        ls1, ls2 = Wfull[f'layer{li}.ls1'], Wfull[f'layer{li}.ls2']
        # phase 1: q/k/v on student-H -> teacher-Q/K/V
        Wm = {}
        C_stud, H_stud = [], []
        for img in range(len(fit_rgb)):
            T, Xin = teach[img][li], cur_in[img]
            H = _ln(Xin, n1w, n1b)
            H_stud.append(H)
        for name, tgt in (('q', 'Q'), ('k', 'K'), ('v', 'V')):
            Sxx = np.zeros((385, 385))
            Sxy = np.zeros((385, 384))
            for img in range(len(fit_rgb)):
                Xa = np.concatenate(
                    [H_stud[img], np.ones((H_stud[img].shape[0], 1))], axis=1)
                Sxx += Xa.T @ Xa
                Sxy += Xa.T @ teach[img][li][tgt]
            Wm[name] = rrr_from_covs(Sxx, Sxy, r)
        # student concat through student softmax
        for img in range(len(fit_rgb)):
            Q = _lin(H_stud[img], Wm['q'])
            K = _lin(H_stud[img], Wm['k'])
            V = _lin(H_stud[img], Wm['v'])
            outs = []
            for h in range(6):
                q, k, v = (a[:, h * 64:(h + 1) * 64] for a in (Q, K, V))
                sc = q @ k.T / 8.0
                sc = np.exp(sc - sc.max(axis=1, keepdims=True))
                outs.append(sc / sc.sum(axis=1, keepdims=True) @ v)
            C_stud.append(np.concatenate(outs, axis=1))
        # phase 2: proj on student-C -> teacher-Oattn
        Sxx = np.zeros((385, 385))
        Sxy = np.zeros((385, 384))
        for img in range(len(fit_rgb)):
            Xa = np.concatenate(
                [C_stud[img], np.ones((C_stud[img].shape[0], 1))], axis=1)
            Sxx += Xa.T @ Xa
            Sxy += Xa.T @ teach[img][li]['Oattn']
        Wm['proj'] = rrr_from_covs(Sxx, Sxy, r)
        # student M -> N2; phase 3: mlp1 on student-N2 -> teacher-P1
        N2_stud, M_stud = [], []
        for img in range(len(fit_rgb)):
            Oa = _lin(C_stud[img], Wm['proj'])
            Mp = cur_in[img] + Oa * ls1
            M_stud.append(Mp)
            N2_stud.append(_ln(Mp, n2w, n2b))
        Sxx = np.zeros((385, 385))
        Sxy = np.zeros((385, 1536))
        for img in range(len(fit_rgb)):
            Xa = np.concatenate(
                [N2_stud[img], np.ones((N2_stud[img].shape[0], 1))], axis=1)
            Sxx += Xa.T @ Xa
            Sxy += Xa.T @ teach[img][li]['P1']
        Wm['mlp1'] = rrr_from_covs(Sxx, Sxy, r)
        # student G; phase 4: mlp2 on student-G -> teacher-Y
        Sxx = np.zeros((1537, 1537))
        Sxy = np.zeros((1537, 384))
        G_stud = []
        for img in range(len(fit_rgb)):
            G = _gelu(_lin(N2_stud[img], Wm['mlp1']))
            G_stud.append(G)
            Xa = np.concatenate([G, np.ones((G.shape[0], 1))], axis=1)
            Sxx += Xa.T @ Xa
            Sxy += Xa.T @ teach[img][li]['Y']
        Wm['mlp2'] = rrr_from_covs(Sxx, Sxy, r)
        seq_W[li] = Wm
        # next inputs + drift vs teacher layer-out
        nxt = []
        for img in range(len(fit_rgb)):
            Yp = _lin(G_stud[img], Wm['mlp2'])
            out = M_stud[img] + Yp * ls2
            nxt.append(out)
            T = teach[img][li]
            tout = T['M'] + T['Y'] * ls2
            drift.setdefault(li, []).append(float(np.mean(np.abs(out - tout))))
        cur_in = nxt
    for li, ds in drift.items():
        print(f"seq L{li} student-out drift: mean|diff|={np.mean(ds):.4f}",
              flush=True)
    # install sequential maps + biases, depth-gate
    for li, r in SEQ_CONFIG.items():
        for name, _, _, di, do in MAPS:
            Wrr = seq_W[li][name]
            bb._w[f'layer{li}.{BUF[name]}'].copy_(torch.from_numpy(
                Wrr[:-1].astype(np.float32).T.reshape(
                    orig[f'layer{li}.{BUF[name]}'].shape)).to(device))
            bkey = f'layer{li}.{BUF[name].replace(".weight", ".bias")}'
            if bkey in bb._w:
                bb._w[bkey].copy_(torch.from_numpy(
                    Wrr[-1].astype(np.float32)).to(device))
    cs = []
    for rgb, ref, cid in scenes:
        cfg = copy.deepcopy(dict(SEED_CONFIG))
        pred = run_pipeline(shared, cfg, [rgb])[0]
        cs.append(_corr(pred, ref))
    print(f"[A-seq] mean={np.mean(cs):.5f} min={min(cs):.5f} "
          f"pass={sum(c >= 0.999 for c in cs)}/{len(cs)}", flush=True)
    for k, v in orig.items():
        bb._w[k].copy_(v)


if __name__ == '__main__':
    main()
