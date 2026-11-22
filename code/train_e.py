"""Train one GNN of the size study. usage: train_e.py CUT FOLD L D SPLIT SEED [STEPS=3000]
SPLIT val: trained on the fold's training ontologies minus the tuned ranker's held-out 20% (for size selection);
SPLIT full: trained on all of the fold's training ontologies (for the test). Regime R of Study B (10% of is-a links hidden
on a training snapshot, batch 8, Adam 3e-3, random name withholding), same sampling as train.py.
Writes results/study_e/models/<CUT>_std_<FOLD>_L<L>d<D>_<SPLIT>_s<SEED>.pt"""
import sys, os, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)
from elib import *
CUT, FOLD, L, D, SP, SEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]); STEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 3000
OUT = os.path.join(out_dir("models"), f"{CUT}_std_{FOLD}_{tag(L, D)}_{SP}_s{SEED}.pt")
if os.path.exists(OUT): sys.exit(0)
TRAIN = DATA["folds"][CUT]["std"][FOLD]["train"] if SP == "full" else split(CUT, FOLD)[1]
G = DATA["graphs"][CUT]; rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
SNAP = {n: view_of(G[n]) for n in TRAIN}
print(f"device {DEV}; {CUT} fold {FOLD} {tag(L, D)} {SP} seed {SEED}: {len(TRAIN)} training snapshots", flush=True)
net = NBFX(d=D, L=L).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=3e-3); t0 = time.time(); B = 8
print(f"parameters {sum(p.numel() for p in net.parameters())}", flush=True)
def dropout(N):
    u = rng.random(); p = 0.0 if u < 0.4 else (1.0 if u > 0.9 else rng.random()); return rng.random(N) >= p
for st in range(STEPS):
    n = TRAIN[rng.integers(len(TRAIN))]; v = SNAP[n]; isa = v.isa; m = rng.random(len(isa)) < 0.10
    if m.sum() == 0: continue
    obs, hid = isa[~m], isa[m]; hid = hid[rng.choice(len(hid), min(B, len(hid)), replace=False)]; q = hid[:, 0]; tgt = hid[:, 1]
    sc = net(on_dev(graph(v, obs)), q, v.chans(q, dropout(len(v.ids)))).clone(); sc[torch.arange(len(q), device=DEV), torch.as_tensor(q, device=DEV)] = -1e9
    loss = nn.functional.cross_entropy(sc, torch.as_tensor(tgt, device=DEV)); opt.zero_grad(); loss.backward(); opt.step()
    if st % 500 == 499: print(f"  step {st + 1} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
torch.save({k: x.cpu() for k, x in net.state_dict().items()}, OUT + ".tmp"); os.replace(OUT + ".tmp", OUT); print(f"saved after {time.time() - t0:.0f}s", flush=True)
