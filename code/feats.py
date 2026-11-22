"""Structural features shared by the hybrid heuristic and the tuned ranker: matrices of an ontology (mats) and per-candidate
structural features for a batch of queries (feats)."""
import numpy as np, scipy.sparse as sp
def mats(o, obs):
    N = len(o["ids"]); P = sp.csr_matrix((np.ones(len(obs)), (obs[:, 0], obs[:, 1])), shape=(N, N)); P.data[:] = 1
    rel = np.array([(a, b) for a, _, b in o["rel"]], dtype=int).reshape(-1, 2); Rm = sp.csr_matrix((np.ones(len(rel)), (rel[:, 0], rel[:, 1])), shape=(N, N)); Rm.data[:] = 1
    A = ((P + P.T + Rm + Rm.T) > 0).astype(np.float64).tocsr(); deg = np.asarray(A.sum(1)).ravel()
    depth = np.zeros(N); cur = P.copy(); d = 1; reach = (np.asarray(P.sum(1)).ravel() > 0)
    for _ in range(12):                                   # depth = longest observed is-a chain length upward, capped
        depth += reach; cur = cur @ P; reach = np.asarray(cur.sum(1)).ravel() > 0
        if not reach.any(): break
    return dict(N=N, P=P, R=Rm, A=A, deg=deg, ideg=np.asarray(P.sum(0)).ravel(), odeg=np.asarray(P.sum(1)).ravel(), depth=depth)
def feats(M, q):
    Q = sp.csr_matrix((np.ones(len(q)), (np.arange(len(q)), q)), shape=(len(q), M["N"])); A, P = M["A"], M["P"]
    nbr = Q @ A                                           # neighbours of each query
    cn = (nbr @ A).toarray(); aa = (nbr @ sp.diags(1 / np.log(np.maximum(M["deg"], 2))) @ A).toarray(); ra = (nbr @ sp.diags(1 / np.maximum(M["deg"], 1)) @ A).toarray()
    simq = nbr @ A; simq = simq.tolil(); simq[np.arange(len(q)), q] = 0; simq = simq.tocsr()
    svote = (simq @ P).toarray()                          # classes sharing neighbours with the query vote for their parents
    relpar = (Q @ (M["R"] + M["R"].T) @ P).toarray()      # parents of the query's relation partners
    uncle = (Q @ P @ P.T @ P).toarray() * 0 + (Q @ P @ P @ P.T).toarray()   # siblings of the query's remaining parents
    pop = np.broadcast_to(np.log1p(M["ideg"]), cn.shape)
    F = np.stack([pop, cn, aa, ra, svote, relpar, uncle, np.broadcast_to(M["depth"], cn.shape), np.broadcast_to(np.log1p(M["deg"]), cn.shape),
                  np.broadcast_to(M["depth"], cn.shape) - M["depth"][q][:, None], np.broadcast_to(np.log1p(M["odeg"]), cn.shape)], 2)
    H = F[:, :, :7]; F = np.concatenate([F, H / (H.max(1, keepdims=True) + 1e-9)], 2)   # also each heuristic relative to the query's best candidate
    return F.astype(np.float32)
