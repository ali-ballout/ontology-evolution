"""Train one GNN (Study B). usage: train.py CUT FOLDSET FOLD REGIME SEED [STEPS=3000]
Regimes (3,000 steps, batch 8, Adam 3e-3, random name withholding; history regimes replace half the batches):
  R deletion only (10% of is-a links hidden on the cutoff snapshot of a training ontology)
  M all history (links added between existing classes + new-class placements, r1 -> rC of training ontologies)
  E existing-class history only      N new-class history only
  X existing-class history restricted to links NOT entailed by r1 (mechanistic test)
  NAMES / STRUCT / NOREL  regime R with inputs restricted (names only / structure only / no relation edges)
  RGPU  regime R trained on a CUDA device (DEVICE=cuda), to check how far results depend on the device
Writes results/study_b/models/<CUT>_<FOLDSET>_<FOLD>_<REGIME>_s<SEED>.pt"""
import sys, os, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from lib import *
CUT, FS, FOLD, REG, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]); STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 3000
OUT = os.path.join(out_dir("models"), f"{CUT}_{FS}_{FOLD}_{REG}_s{SEED}.pt")
if os.path.exists(OUT): sys.exit(0)
TRAIN = DATA["folds"][CUT][FS][FOLD]["train"]; G = DATA["graphs"][CUT]; mode = mode_of(REG); rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
SNAP = {n: view_of(G[n], mode=mode) for n in TRAIN}; HIST = {}
if REG in ("M", "E", "N", "X"):
    HI = DATA["history"].get(CUT, {})
    for n in TRAIN:
        if n not in HI: continue
        h = HI[n]; A = [(a["x"], a["y"]) for a in h["A"] if REG != "X" or not a["entailed"]]; B = [(x, p) for x, b in h["B"].items() for p in b["parents"]]
        ex = {"M": A + B, "E": A, "X": A, "N": B}[REG]
        if len(ex) < 5: continue
        g1 = DATA["graphs"][h["r1"]][n]; v = view_of(g1, [(x, b["name"]) for x, b in h["B"].items()], mode)
        HIST[n] = dict(v=v, g=graph(v), N0=len(g1["ids"]), ex=ex, par=h["par"])
DEV = torch.device(os.environ.get("DEVICE", "cpu"))
for h in HIST.values(): h["g"] = to_device(h["g"], DEV)
print(f"{CUT} {FS} fold {FOLD} regime {REG} seed {SEED}: {len(TRAIN)} training snapshots, {len(HIST)} with history ({sum(len(h['ex']) for h in HIST.values())} examples)", flush=True)
if REG in ("M", "E", "N", "X") and not HIST: print("no history examples; nothing to train", flush=True); sys.exit(2)
net = NBF().to(DEV); opt = torch.optim.Adam(net.parameters(), lr=3e-3); t0 = time.time(); HN = sorted(HIST); B = 8; GC = {}
def dropout(N):
    u = rng.random(); p = 0.0 if u < 0.4 else (1.0 if u > 0.9 else rng.random()); return rng.random(N) >= p
for st in range(STEPS):
    if HN and rng.random() < 0.5:
        n = HN[rng.integers(len(HN))]; h = HIST[n]; v = h["v"]; pick = [h["ex"][i] for i in rng.choice(len(h["ex"]), min(B, len(h["ex"])), replace=False)]
        q = np.array([v.ix[c] for c, _ in pick]); tgt = np.array([v.ix[p] for _, p in pick])
        sc = net(h["g"], torch.tensor(q, device=DEV), v.chans(q, dropout(len(v.ids)))).clone(); sc[:, h["N0"]:] = -1e9; sc[torch.arange(len(q), device=DEV), torch.tensor(q, device=DEV)] = -1e9
        for i, (c, p) in enumerate(pick):
            oth = [v.ix[x] for x in h["par"].get(c, []) if x in v.ix and x != p]
            if oth: sc[i, oth] = -1e9
    else:
        n = TRAIN[rng.integers(len(TRAIN))]; v = SNAP[n]; isa = v.isa; m = rng.random(len(isa)) < 0.10
        if m.sum() == 0: continue
        obs, hid = isa[~m], isa[m]; hid = hid[rng.choice(len(hid), min(B, len(hid)), replace=False)]; q = hid[:, 0]; tgt = hid[:, 1]
        sc = net(to_device(graph(v, obs), DEV), torch.tensor(q, device=DEV), v.chans(q, dropout(len(v.ids)))).clone(); sc[torch.arange(len(q), device=DEV), torch.tensor(q, device=DEV)] = -1e9
    loss = nn.functional.cross_entropy(sc, torch.tensor(tgt, device=DEV)); opt.zero_grad(); loss.backward(); opt.step()
    if st % 500 == 499: print(f"  step {st + 1} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
torch.save(net.to("cpu").state_dict(), OUT + ".tmp"); os.replace(OUT + ".tmp", OUT); print("saved", flush=True)
