# coding: utf-8
r"""VISA (standalone): Vicinity-Integrated Spectral Aggregation.

     Look for "def __init__" and search the four modules:
     -- IGF : item-side graph filter (frozen MM-GF baseline, compact spec)
     -- SBG : spectral-balanced graph construction (band-equalized cosine)
     -- VSP : vicinity score propagation over the LLM-persona user graph
     -- MARF: magnitude-aligned residual fusion of s_item + s_user

Model kinds each mix the four modules; final = MARF(IGF, VSP(SBG(profile))).
Code name was MMGFUU (prior layout: MM-GF item side + user-user residual);
renamed to VISA 2026-09-28 for module naming. Self-contained (no MMGF import).

IGF, item-side graph filter (FROZEN, bit-for-bit MM-GF baseline, 0.1032):
  R~ = D_r^{-alpha} R D_c^{alpha-1}
  Pbar = (R~^T R~)^{o adjust_s}
  P_ui = a1.P + a2.P^2 + a3.P^3   (lmin-shift + lambda* normalization)
  modality (MBG): cosine -> top-k binary -> (t/v NCER) -> Eq.10 norm -> poly[1]
  Pmm = P_ui + beta.P_txt + gamma.P_img
  scores_item = R @ Pmm            (batch SpMV, M x N)

SBG, spectral-balanced graph (novel):
  Motivation: cosine similarity on raw features is dominated by LOW-FREQ
  energy (1/f spectrum), high-freq details collapse as O(1/sqrt(rho)).
  Solution:
    F -> SVD basis V (cols sorted by singular value desc = freq asc)
      -> project F_proj = F @ V
      -> split into B bands (low / mid / high)
      -> per-band L2 normalize (equalize energy across bands)
      -> scale by sqrt(w_b), concat, global L2 normalize
      -> cosine top-k

  Band boundaries:
    - Default (legacy): uniform by dimension count.
    - NEW (Scheme A): energy-adaptive. `sbg_band_edges` is a comma list of
      cumulative energy ratios (in (0,1], non-decreasing). Each band b
      holds components until cumulative energy reaches sbg_band_edges[b].
      This isolates the TRUE mid-frequency region under 1/f spectra.
      Example B=3: "0.7,0.95,1.0" -> low 70% energy, mid 25%, high 5%.

  Config:
    sbg_enabled     : master switch (default False -> bit-for-bit baseline)
    sbg_bands       : B, number of bands (default 3)
    sbg_init_w      : per-band weights, comma list (mean-normalized to 1);
                      treated as hyperparameters grid-searched on valid
    sbg_uu          : also apply SBG to VSP graph construction (default False)
    sbg_band_edges  : NEW, comma list of cumulative energy ratios in (0,1];
                      empty/missing -> uniform dimension split (legacy)
    sbg_whiten      : use PCA whitening (divide each PC by sv^pow) instead of
                      per-band L2 balancing; sbg_whiten_pow/rank/eps control it

VSP, vicinity score propagation (novel; active only when mu_user > 0):
  H_U = self.u_feat (M x d, from user_feat.npy)          (LLM persona, LPP)
  S = top-k(uu_k) cosine user graph, no self-loop, D^{-1/2} S D^{-1/2}
  scores_user = S_norm @ scores_item  (== (S@R)@Pmm)
  row-RMS normalize (v_norm substep)
  (2026-09-28 simplification: train-item masking, per-user degree weight,
   and directed row-stochastic variant measured noise-level across
   baby/sports/clothing -> removed from default pipeline)

MARF, magnitude-aligned residual fusion (novel):
  scale = RMS(item)/RMS(uu), wt = mu_user (const)
  final = scores_item + mu_user*scale*scores_user

mu_user=0 reproduces the item baseline bit-for-bit.
Legacy experiment codes (v1/v2/.../v8, scheme A/C/D) verified useless;
only the single-path pipeline above is kept.
"""

import os
import time
import zlib
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class VISA(GeneralRecommender):
    def __init__(self, config, dataset):
        super(VISA, self).__init__(config, dataset)

        # ---------- item-side hyperparams ----------
        self.alpha = float(config['alpha'])
        self.adjust_s = float(config['adjust_s'])
        self.a1 = float(config['a1'])
        self.a2 = float(config['a2'])
        self.a3 = float(config['a3'])
        self.beta = float(config['beta'])
        self.gamma = float(config['gamma'])
        self.knn_k = int(config['knn_k'])
        self.ncer = bool(config['ncer']) if 'ncer' in config else False
        self.ncer_pow = float(config['ncer_pow']) if 'ncer_pow' in config else 0.0

        # ---------- SBG (Spectral-Balanced Graph) ----------
        self.sbg_enabled = bool(config['sbg_enabled']) if 'sbg_enabled' in config else False
        self.sbg_bands = int(config['sbg_bands']) if 'sbg_bands' in config else 3
        self.sbg_uu = bool(config['sbg_uu']) if 'sbg_uu' in config else False
        self.sbg_mode = str(config['sbg_mode']) if 'sbg_mode' in config else 'adaptive'
        # design-choice ablation: keep truncated-SVD projection but skip the
        # per-band L2 re-weighting (paper's "truncated SVD only" row)
        self.sbg_svd_only = bool(config['sbg_svd_only']) if 'sbg_svd_only' in config else False
        # SBG variant: PCA whitening instead of per-band L2 balancing.
        #   sbg_whiten      : use whitening (overrides band weighting)
        #   sbg_whiten_pow  : exponent on singular values (1.0 = full whitening,
        #                     0.5 = sqrt/mild whitening, 0.0 = plain rotation)
        #   sbg_whiten_rank : keep only the top-R principal components (0 = all)
        #   sbg_whiten_eps  : floor on singular values before division
        self.sbg_whiten = bool(config['sbg_whiten']) if 'sbg_whiten' in config else False
        self.sbg_whiten_pow = float(config['sbg_whiten_pow']) if 'sbg_whiten_pow' in config else 1.0
        self.sbg_whiten_rank = int(config['sbg_whiten_rank']) if 'sbg_whiten_rank' in config else 0
        self.sbg_whiten_eps = float(config['sbg_whiten_eps']) if 'sbg_whiten_eps' in config else 1e-8
        self._sbg_pca = {}                                   # cached PCA bases
        self._sbg_sv = {}                                    # cached singular values
        self._sbg_bounds = {}                                # cached band boundaries
        self.dataset_name = config['dataset']
        # PCA backend: 'eigh' (Gram-matrix eigh on GPU, exact, default),
        # 'svd_lowrank' (legacy randomized), 'svd' (economy full SVD).
        self.pca_backend = str(config['pca_backend']) if 'pca_backend' in config else 'eigh'
        # Disk cache for (V, sv): the basis only depends on the static feature
        # matrix, so it can be precomputed once and reused across runs.
        self.pca_cache = bool(config['pca_cache']) if 'pca_cache' in config else True
        if os.environ.get('VISA_PCA_CACHE', '1') == '0':
            self.pca_cache = False
        self.pca_cache_dir = os.environ.get(
            'VISA_PCA_CACHE_DIR',
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), 'cache', 'sbg_pca'))

        # ---------- stage timing / peak memory (experiment C) ----------
        self._stage_t = {'SVD': 0.0, 'topk+NCER': 0.0,
                         'power_iter': 0.0, 'SpMM': 0.0, 'UU': 0.0}
        self._rank_t = {'t': 0.0, 'n': 0}

        # ---- NEW: 能量自适应频带边界 ----
        self.sbg_band_edges = None

        def _float_list(v):
            """parse '0.7,0.95,1.0' / (0.7,0.95,1.0) / [0.7,0.95,1.0] -> floats"""
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
            s = str(v).strip().strip('[]()').strip()
            if not s:
                return None
            return [float(x) for x in s.split(',')]

        #读取sbg相关参与，预计算PCA基
        if self.sbg_enabled:
            _edges_str = config['sbg_band_edges'] if 'sbg_band_edges' in config else ''
            # sbg_mode: 'adaptive' (per-dataset) or 'equal' (parameter-free fixed)
            if getattr(self, 'sbg_mode', 'adaptive') == 'equal':
                # parameter-free fixed recipe: fixed energy cut + equal weights
                if self.sbg_bands == 3:
                    self.sbg_band_edges = [0.70, 0.95, 1.0]
                else:
                    # fallback: uniform-ish energy split
                    edges = []
                    for i in range(1, self.sbg_bands):
                        edges.append(i / self.sbg_bands)
                    edges = [e * 0.95 + 0.05 / self.sbg_bands for e in edges]
                    self.sbg_band_edges = [round(x, 6) for x in edges] + [1.0]
                init_w = [1.0] * self.sbg_bands
                if 'sbg_init_w' in config and config['sbg_init_w']:
                    try:
                        _w = _float_list(config['sbg_init_w'])
                        if _w and len(_w) == self.sbg_bands:
                            init_w = _w
                    except (ValueError, TypeError):
                        pass
            elif _edges_str:
                self.sbg_band_edges = _float_list(_edges_str)
                assert len(self.sbg_band_edges) == self.sbg_bands, \
                    'sbg_band_edges length must equal sbg_bands'
                assert all(0.0 < e <= 1.0 for e in self.sbg_band_edges), \
                    'sbg_band_edges must be in (0, 1]'
                assert all(self.sbg_band_edges[i] <= self.sbg_band_edges[i + 1]
                           for i in range(len(self.sbg_band_edges) - 1)), \
                    'sbg_band_edges must be non-decreasing'
                print('[SBG] energy-adaptive band edges: {}'.format(
                    self.sbg_band_edges), flush=True)
                # weights
                if 'sbg_init_w' in config and config['sbg_init_w']:
                    init_w = _float_list(config['sbg_init_w'])
                else:
                    init_w = [1.0 / (1.0 + 0.5 * i) for i in range(self.sbg_bands)]
            else:
                print('[SBG] sbg_band_edges not set -> uniform dimension split',
                      flush=True)
                if 'sbg_init_w' in config and config['sbg_init_w']:
                    init_w = _float_list(config['sbg_init_w'])
                else:
                    init_w = [1.0 / (1.0 + 0.5 * i) for i in range(self.sbg_bands)]
            assert len(init_w) == self.sbg_bands, 'sbg_init_w length != sbg_bands'
            init_w_t = torch.tensor(init_w, dtype=torch.float32)
            init_w_t = init_w_t / init_w_t.mean().clamp(min=1e-12)
            # band weights are hyperparameters (grid-searched on valid),
            # never learned (legacy sbg_learn_w/sbg_w_reg were dead params,
            # removed 2026-09-29; training is disabled so nothing is trained)
            self.register_buffer('_sbg_w_fixed', init_w_t)

            # precompute PCA bases for each modality
            _t_svd = time.time()
            if self.t_feat is not None:
                _F = self.t_feat.detach().cpu().numpy()
                self._compute_pca_basis(_F, 'txt')
                print('[SBG] txt PCA basis: {} -> {}'.format(_F.shape,
                      tuple(self._sbg_pca['txt'].shape)), flush=True)
            if self.v_feat is not None:
                _F = self.v_feat.detach().cpu().numpy()
                self._compute_pca_basis(_F, 'img')
                print('[SBG] img PCA basis: {} -> {}'.format(_F.shape,
                      tuple(self._sbg_pca['img'].shape)), flush=True)
            if self.u_feat is not None:
                _F = self.u_feat.detach().cpu().numpy()
                self._compute_pca_basis(_F, 'user')
                print('[SBG] user PCA basis: {} -> {}'.format(_F.shape,
                      tuple(self._sbg_pca['user'].shape)), flush=True)
            self._stage_t['SVD'] += time.time() - _t_svd

            print('[SBG] enabled B={} init_w={} edges={}'.format(
                self.sbg_bands,
                init_w_t.tolist(), self.sbg_band_edges), flush=True)

        # ---------------- item side (FROZEN MM-GF baseline) ----------------
        _t = time.time()
        R = dataset.inter_matrix(form='coo').astype(np.float64)
        dr = np.asarray(R.sum(axis=1)).ravel()
        dc = np.asarray(R.sum(axis=0)).ravel()
        dr[dr > 0] = np.power(dr[dr > 0], -self.alpha)
        dc[dc > 0] = np.power(dc[dc > 0], self.alpha - 1)
        Rn = R.copy()
        Rn.data *= dr[Rn.row] * dc[Rn.col]
        self.Rn = Rn
        self.Rr = R.copy()

        _R32 = Rn.astype(np.float32)
        Pbar_ui_f = self._poly_filter(
            self._hadamard_power((_R32.T @ _R32).toarray(), self.adjust_s),
            'UI', [self.a1, self.a2, self.a3])
        del _R32
        self._stage_t['power_iter'] += time.time() - _t

        _t = time.time()
        self._modality_filter = self._mk_modality_filter()

        Pmm = Pbar_ui_f.detach().cpu()
        if self.t_feat is not None:
            Pmm = Pmm + self.beta * self._modality_filter(self.t_feat,
                                                          self.ncer, 'txt').detach().cpu()
        if self.v_feat is not None:
            Pmm = Pmm + self.gamma * self._modality_filter(self.v_feat,
                                                           self.ncer, 'img').detach().cpu()
        del Pbar_ui_f
        self._stage_t['topk+NCER'] += time.time() - _t

        _t = time.time()
        scores_item = torch.zeros(self.n_users, self.n_items,
                                  device=self.device, dtype=torch.float32)
        Pmm = Pmm.to(self.device)
        r_csr = self.Rr.tocsr()
        for i0 in range(0, self.n_users, 512):
            i1 = min(i0 + 512, self.n_users)
            rows = r_csr[i0:i1]
            idx = torch.from_numpy(np.stack(rows.nonzero())).to(self.device).long()
            if idx.numel() == 0:
                continue
            val = torch.from_numpy(rows.data.astype(np.float32)).to(self.device)
            Rb = torch.sparse_coo_tensor(
                torch.stack([idx[0] % (i1 - i0), idx[1]]), val,
                size=(i1 - i0, self.n_items))
            scores_item[i0:i1] = torch.sparse.mm(Rb, Pmm)
        del Pmm, r_csr
        self.scores_matrix = scores_item
        self._stage_t['SpMM'] += time.time() - _t

        # ---------------- UU side (optional, single pipeline) ----------------
        mu_user = float(config['mu_user']) if 'mu_user' in config else 0.0
        if mu_user <= 0:
            # peak gpu memory (model-build phase) + stage timings still reported
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
            print('[VISA-SYS] stage_times(s)={} peak_gpu_gb={:.3f}'.format(
                {k: round(v, 4) for k, v in self._stage_t.items()}, peak_mem),
                flush=True)
            return                                         # pure item baseline
        uu_k = int(config['uu_k']) if 'uu_k' in config else 20
        uu_ab = config['uu_ablate'] or ''
        uu_src = config['uu_src'] if 'uu_src' in config else 'profile'
        # MARF / VSP design-choice ablation switches (all default = off)
        uu_no_rms = bool(config['uu_no_rms']) if 'uu_no_rms' in config else False
        uu_no_scale = bool(config['uu_no_scale']) if 'uu_no_scale' in config else False
        uu_prop_raw = bool(config['uu_prop_raw']) if 'uu_prop_raw' in config else False

        _t_uu = time.time()
        print('[UU-Filtering] mu={} uu_k={} uu_src={} ablate="{}" sbg_uu={}'
              ' no_rms={} no_scale={} prop_raw={} ...'.format(
                  mu_user, uu_k, uu_src, uu_ab, self.sbg_uu,
                  uu_no_rms, uu_no_scale, uu_prop_raw), flush=True)

        if uu_src in ('random', 'cooccur'):
            Sp = None                                       # built below
        elif uu_src == 'shuffle':
            # negative control: row-permute the LLM persona features (same
            # marginal distribution, semantics destroyed). Fixed RNG seed.
            torch.manual_seed(42)
            perm = torch.randperm(self.u_feat.size(0))
            Sp = self._mk_graph(
                self.u_feat[perm], uu_k, pca_key='user',
                use_sbg=self.sbg_uu,
                exclude_self=False, symmetrize=False,
                ncer=self.ncer, norm='bilateral', out='sparse',
            )
        else:                                               # 'profile' (default)
            #propagate over scores_item
            Sp = self._mk_graph(
                self.u_feat, uu_k, pca_key='user',
                use_sbg=self.sbg_uu,
                exclude_self=False, symmetrize=False,
                ncer=self.ncer, norm='bilateral', out='sparse',
            )
        if uu_src == 'random':
            # degree-matched uniform random neighbors (fixed seed), same
            # bilateral normalization as the cosine pipeline
            Sp = self._mk_random_graph(uu_k)
        elif uu_src == 'cooccur':
            # first-order co-occurrence user graph: R_train R_train^T
            Sp = self._mk_cooccur_graph(uu_k)
        if uu_prop_raw:
            # design-choice row: propagate raw interactions R instead of the
            # learned item scores matrix (scores_user = S_U R)
            Rcoo = self.Rr.tocoo()
            r_idx = torch.from_numpy(np.stack([Rcoo.row, Rcoo.col])).to(self.device).long()
            r_val = torch.from_numpy(Rcoo.data.astype(np.float32)).to(self.device)
            Rsp = torch.sparse_coo_tensor(
                r_idx, r_val, (self.n_users, self.n_items)).coalesce()
            scores_user = Sp.mm(Rsp).to_dense()
        else:
            scores_user = Sp.mm(scores_item)
        #row-RMS normalize (design-choice: can be disabled)
        if not uu_no_rms:
            rms_u = torch.sqrt(torch.mean(scores_user ** 2, dim=1,
                                            keepdim=True)).clamp(min=1e-12)
            scores_user = scores_user / rms_u
        # fusion: magnitude alignment + constant per-user weight
        if uu_no_scale:
            scale = 1.0                                     # w/o global scale ratio
        else:
            scale = torch.sqrt(torch.mean(scores_item ** 2)) / (
                torch.sqrt(torch.mean(scores_user ** 2)) + 1e-12)

        wt = mu_user
        self.scores_matrix = scores_item + (wt * scale) * scores_user
        self._stage_t['UU'] += time.time() - _t_uu
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print('[VISA] fused (MARF; VSP rms) mu={} | UU {:5.2f}s | '
              'scale={:.4f}'.format(mu_user, self._stage_t['UU'], float(scale)),
              flush=True)
        print('[VISA-SYS] stage_times(s)={} peak_gpu_gb={:.3f}'.format(
            {k: round(v, 4) for k, v in self._stage_t.items()}, peak_mem),
            flush=True)

    # ======================================================================
    # SBG helpers
    # ======================================================================
    def _pca_cache_path(self, F_np, key):
        """Content-addressed cache file for (V, sv) of a feature matrix."""
        if not self.pca_cache:
            return None
        crc = zlib.crc32(np.ascontiguousarray(F_np).view(np.uint8)) & 0xFFFFFFFF
        tag = '{}_{}_{}x{}_{}_{:08x}'.format(
            self.dataset_name, key, F_np.shape[0], F_np.shape[1],
            self.pca_backend, crc)
        return os.path.join(self.pca_cache_dir, tag + '.pt')

    def _pca_svd(self, F_np, key):
        """Right singular vectors V (d, r) sorted by descending singular value.

        Backends:
          eigh        : V = eigvec(F^T F) on GPU (exact, O(N d^2 + d^3));
                        sigma = sqrt(lambda). Fastest by far when d << N.
          svd_lowrank : legacy randomized SVD (kept for reproducibility).
          svd         : economy torch.linalg.svd.
        """
        F_t = torch.from_numpy(np.asarray(F_np, dtype=np.float32))
        N, d = F_t.shape
        r = min(N, d)
        backend = self.pca_backend
        if backend == 'eigh' and d <= 8192 and N >= d:
            Fg = F_t.to(self.device)
            C = Fg.T @ Fg                                     # (d, d) Gram
            del Fg
            evals, evecs = torch.linalg.eigh(C)
            del C
            evals = evals.flip(0).clamp(min=0.0)              # descending
            V = evecs.flip(1)                                 # (d, d)
            sv = evals.sqrt()
            return V.cpu(), sv.cpu()
        if backend == 'svd' or (backend == 'eigh' and d <= 2048):
            _, S, Vh = torch.linalg.svd(F_t, full_matrices=False)
            return Vh.T.cpu(), S.cpu()
        torch.manual_seed(42)
        _, S, V = torch.svd_lowrank(F_t, q=r, niter=4)
        perm = torch.argsort(S, descending=True)
        return V[:, perm].cpu(), S[perm].cpu()

    def _set_band_bounds(self, key, sv):
        """Energy-adaptive band boundaries from singular values (cached)."""
        if self.sbg_band_edges is None or sv is None:
            return
        energy = (sv.double() ** 2)                          # per-component energy
        total = energy.sum().clamp(min=1e-30)
        cum = torch.cumsum(energy, dim=0) / total            # monotone -> 1.0
        r_eff = sv.numel()
        bounds = [0]
        for e in self.sbg_band_edges:
            idx = int(torch.searchsorted(
                cum, torch.tensor(float(e), dtype=cum.dtype)).item())
            idx = min(idx + 1, r_eff)
            bounds.append(idx)
        for i in range(1, len(bounds)):
            if bounds[i] < bounds[i - 1]:
                bounds[i] = bounds[i - 1]
        bounds[-1] = r_eff                                   # tail aligned to r
        self._sbg_bounds[key] = bounds
        cum_at = [round(float(cum[b - 1]), 4) if b > 0 else 0.0
                  for b in bounds[1:]]
        print('[SBG] {} band bounds (idx)={} | cum_energy={}'.format(
            key, bounds, cum_at), flush=True)

    def _compute_pca_basis(self, F_np, key):
        """Compute PCA basis V (d, r), columns sorted by singular value desc.
        Column order corresponds to frequency: earliest = lowest freq
        (largest variance), latest = highest freq (smallest variance).

        Additionally, if sbg_band_edges is set, computes and caches the
        energy-adaptive band boundaries under `key`.
        """
        F_np = np.asarray(F_np, dtype=np.float32)
        path = self._pca_cache_path(F_np, key)
        if path is not None and os.path.isfile(path):
            try:
                blob = torch.load(path, map_location='cpu')
                self._sbg_pca[key] = blob['V'].float()
                self._sbg_sv[key] = blob['sv'].float()
                self._set_band_bounds(key, self._sbg_sv[key])
                print('[SBG] {} PCA basis (disk cache): {}'.format(
                    key, path), flush=True)
                return self._sbg_pca[key]
            except Exception as e:
                print('[SBG] cache load failed ({}); recomputing'.format(e),
                      flush=True)

        try:
            V, sv = self._pca_svd(F_np, key)
        except Exception as e:
            print('[SBG] SVD failed ({}); fallback to identity basis'.format(e),
                  flush=True)
            V = torch.eye(F_np.shape[1])
            sv = torch.ones(min(F_np.shape))

        self._sbg_pca[key] = V
        self._sbg_sv[key] = sv.detach().float()
        self._set_band_bounds(key, self._sbg_sv[key])

        if path is not None:
            try:
                os.makedirs(self.pca_cache_dir, exist_ok=True)
                tmp = path + '.tmp%d' % os.getpid()
                torch.save({'V': self._sbg_pca[key], 'sv': self._sbg_sv[key]}, tmp)
                os.replace(tmp, path)
            except Exception as e:
                print('[SBG] cache write failed ({}); continue'.format(e),
                      flush=True)

        return self._sbg_pca[key]

    def _get_sbg_w(self):
        """Current SBG band weights (B,), mean = 1 (fixed hyperparameters)."""
        if self.sbg_enabled:
            return self._sbg_w_fixed
        else:
            return torch.ones(self.sbg_bands)

    def _sbg_normalize(self, features, pca_key):
        """Spectral-balanced row-L2-normalized features.

        Steps:
          1. project F -> F @ V (freq-sorted columns)
          2. split columns into B bands:
             - if sbg_band_edges is set: energy-adaptive bounds (cached)
             - otherwise: uniform dimension split (legacy)
          3. per-band L2 normalize each row
          4. scale each band by sqrt(w_b)
          5. global row L2 normalize
        """
        if isinstance(features, np.ndarray):
            F_t = torch.from_numpy(features).float()
        else:
            F_t = features.detach().float()

        V = self._sbg_pca.get(pca_key, None)
        if V is None:
            # defensive fallback: no cached basis (should not happen in
            # practice, since txt/img/user bases are precomputed in __init__)
            # -> plain cosine, no band equalization
            return F.normalize(F_t, p=2, dim=-1)
        V = V.to(F_t.device)

        F_proj = F_t @ V                                       # (N, r)
        N, r = F_proj.shape

        # ---- PCA whitening variant (replaces band balancing) ----
        if self.sbg_whiten:
            sv = self._sbg_sv.get(pca_key, None)
            if sv is None:
                return F.normalize(F_proj, p=2, dim=-1)
            R = sv.numel() if self.sbg_whiten_rank <= 0 \
                else min(int(self.sbg_whiten_rank), sv.numel())
            F_proj = F_proj[:, :R]
            sv = sv[:R].to(F_proj.device).clamp(min=self.sbg_whiten_eps)
            scale = sv.pow(-self.sbg_whiten_pow)              # (R,)
            return F.normalize(F_proj * scale, p=2, dim=-1)

        B = self.sbg_bands
        if r < B:
            # too few components for the requested band count
            return F.normalize(F_t, p=2, dim=-1)

        # ---- 选择边界方案 ----
        starts = None
        if self.sbg_band_edges is not None and pca_key in self._sbg_bounds:
            cached = self._sbg_bounds[pca_key]
            if len(cached) == B + 1:
                starts = cached

        if starts is None:
            # legacy: uniform dimension split
            sizes = [r // B] * B
            for i in range(r % B):
                sizes[i] += 1
            starts = [0]
            for s in sizes:
                starts.append(starts[-1] + s)

        w = self._get_sbg_w().to(F_t.device)
        if self.sbg_svd_only:
            # design-choice row: truncated-SVD projection only, no per-band
            # L2 re-weighting (paper's "truncated SVD only")
            return F.normalize(F_proj, p=2, dim=-1)

        sqrt_w = torch.sqrt(w.clamp(min=0.0))

        # 用 zeros 初始化，容忍空频带
        F_balanced = torch.zeros_like(F_proj)
        for b in range(B):
            s, e = starts[b], starts[b + 1]
            if e <= s:                                         # 空频带，跳过
                continue
            F_b = F_proj[:, s:e]
            nb = F_b.norm(dim=1, keepdim=True).clamp(min=1e-12)
            F_balanced[:, s:e] = sqrt_w[b] * (F_b / nb)

        return F.normalize(F_balanced, p=2, dim=-1)

    # ======================================================================
    # Item side helpers
    # ======================================================================
    def _mk_modality_filter(self):
        """Return fn(feat, ncer, pca_key) building a modality graph -> poly filter."""
        def filt(feat, ncer, pca_key='txt'):
            Sn = self._mk_graph(
                feat, self.knn_k, pca_key=pca_key,
                exclude_self=False, symmetrize=False,
                ncer=ncer, norm='bilateral', out='csr',
            )
            return self._poly_filter(
                self._hadamard_power((Sn @ Sn.T).toarray(), self.adjust_s),
                'mod', [1.0])
        return filt

    def _mk_graph(self, features, k, pca_key='txt', use_sbg=None,
                exclude_self=False, symmetrize=False, ncer=False,
                norm='bilateral', out='csr'):
        """统一的 top-k 余弦图构建器（物品模态侧与用户侧共用）。

        Args:
            features:     (N, d)
            k:            top-k 邻居数
            pca_key:      SBG PCA 基键（'txt' | 'img' | 'user'）
            use_sbg:      是否用 SBG 归一化；None -> 跟随 self.sbg_enabled
            exclude_self: 是否排除自身（用户图 True，物品图 False）
            symmetrize:   是否并集对称化（用户图 True，物品图 False）
            ncer:         是否做 NCER 三角提纯（仅物品侧）
            norm:         'bilateral' -> D_r^-a S D_c^(a-1) 非对称
                        'symmetric' -> D^-1/2 S D^-1/ 对称
            out:          'csr'    -> scipy.sparse.csr_matrix
                        'sparse' -> torch.sparse_coo_tensor

        Returns:
            归一化后的图，格式由 out 决定。
        """
        # ---- 1. 特征归一化 ----
        if use_sbg is None:
            use_sbg = self.sbg_enabled
        if use_sbg and self.sbg_enabled:
            Fn = self._sbg_normalize(features, pca_key)
        else:
            Fn = F.normalize(features.detach().float(), p=2, dim=-1)

        n, dev = Fn.size(0), Fn.device
        k = min(int(k), n - 1)

        # ---- 2. 分块 top-k ----
        cols = []
        for i0 in range(0, n, 2048):
            i1 = min(i0 + 2048, n)
            sim = Fn[i0:i1] @ Fn.T
            if exclude_self:
                sim = sim.clamp_(min=0.0)
                sim[torch.arange(i1 - i0, device=dev),
                    torch.arange(i0, i1, device=dev)] = -1e9
            cols.append(sim.topk(k, dim=-1).indices)
        col = torch.cat(cols)
        row = torch.arange(n, device=dev).unsqueeze(1).expand_as(col)

        # ---- 3. 二值图（可选对称化） ----
        r, c = row.reshape(-1), col.reshape(-1)
        if symmetrize:
            idx = torch.stack([torch.cat([r, c]), torch.cat([c, r])])
        else:
            idx = torch.stack([r, c])
        S = torch.sparse_coo_tensor(
            idx, torch.ones(idx.size(1), device=dev), (n, n)
        ).coalesce()
        S = torch.sparse_coo_tensor(                             # 二值化
            S.indices(), torch.ones_like(S.values()), S.size()
        ).coalesce()

        # ---- 4. NCER（仅物品侧，稀疏版：只在已有边上算 Jaccard） ----
        if ncer:
            S = self._ncer_sparse(S)

        # ---- 5. 归一化（COO 上按度数缩放，不再 densify） ----
        if norm == 'bilateral':
            Sn = self._bilateral_coo(S)
            if out == 'csr':
                coo = Sn.coalesce()
                ii = coo.indices().cpu().numpy()
                return sp.csr_matrix(
                    (coo.values().cpu().numpy(), (ii[0], ii[1])),
                    shape=Sn.shape)
            return Sn

        elif norm == 'symmetric':
            deg = torch.sparse.sum(S, dim=1).to_dense().clamp_(min=1.0)
            d = deg.pow(-0.5)
            idx = S.indices()
            Sn = torch.sparse_coo_tensor(
                idx, d[idx[0]] * d[idx[1]], S.size()
            ).coalesce()
            if out == 'sparse':
                return Sn
            Sn = Sn.cpu().coalesce()
            coo = Sn.indices().numpy()
            return sp.csr_matrix(
                (Sn.values().numpy(), (coo[0], coo[1])), shape=Sn.shape)

        raise ValueError('unknown norm: {}'.format(norm))

    def _mk_random_graph(self, k):
        """Degree-matched uniform random neighbor graph (fixed seed 42).

        Each user u picks exactly k distinct random neighbors (excluding
        self), then applies the same bilateral degree normalization used by
        the cosine pipeline. Used as the 'Random' user-graph source.
        """
        m = self.n_users
        dev = self.device
        torch.manual_seed(42)
        k = min(int(k), m - 1)
        # per-row random permutation of all-but-self indices
        perm = torch.randperm(m - 1, device=dev)
        base = perm[:k].unsqueeze(0).expand(m, k).clone()
        shift = torch.arange(m, device=dev)
        # convert index in [0, m-2] to index in [0, m-1] excluding self
        src = base + (shift[:, None] >= base).long()

        # directed graph (out-edges only), binary
        r = torch.arange(m, device=dev).unsqueeze(1).expand_as(src).reshape(-1)
        c = src.reshape(-1)
        S = torch.sparse_coo_tensor(
            torch.stack([r, c]), torch.ones(r.size(0), device=dev),
            (m, m)).coalesce()
        S = torch.sparse_coo_tensor(
            S.indices(), torch.ones_like(S.values()), S.size()).coalesce()
        return self._bilateral_sparse(S)

    def _mk_cooccur_graph(self, k):
        """First-order co-occurrence user graph R_train R_train^T.

        Row-normalized to [0,1], per-user top-k, then bilateral degree
        normalization. Used as the 'cooccur' user-graph source.
        """
        Rcoo = self.Rr.tocoo()
        Rsp = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([Rcoo.row, Rcoo.col])).long(),
            torch.from_numpy(Rcoo.data.astype(np.float32)),
            (self.n_users, self.n_items))
        # row-normalize
        deg = torch.sparse.sum(Rsp, dim=1).to_dense().clamp_(min=1.0)
        idx = Rsp.indices()
        val = Rsp.values() / deg[idx[0]]
        Rn = torch.sparse_coo_tensor(idx, val, Rsp.size())
        C = torch.sparse.mm(Rn, Rn.transpose(0, 1)).to_dense().cpu().numpy()
        k = min(int(k), self.n_users - 1)
        topk = np.argpartition(-C, kth=k - 1, axis=1)[:, :k]
        topk = topk.astype(np.int64)
        n = self.n_users
        rows = np.repeat(np.arange(n), k)
        cols = topk.reshape(-1)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        S = torch.sparse_coo_tensor(
            torch.stack([torch.from_numpy(rows).long(),
                         torch.from_numpy(cols).long()]),
            torch.ones(rows.size, dtype=torch.float32), (n, n)).coalesce()
        return self._bilateral_sparse(S)

    def _bilateral_sparse(self, S):
        """D_r^-alpha S D_c^(alpha-1) bilateral normalization (out='sparse')."""
        return self._bilateral_coo(S)

    def _bilateral_coo(self, S):
        """Bilateral degree normalization directly on COO values (no densify).

        Equivalent to D_r^-alpha S D_c^(alpha-1) where the degrees are the
        (binary) out-degrees used by the previous dense implementation.
        """
        S = S.coalesce()
        idx = S.indices()
        v = S.values().float()
        n = S.size(0)
        dr = torch.zeros(n, device=v.device, dtype=v.dtype).scatter_add_(
            0, idx[0], v)
        dc = torch.zeros(n, device=v.device, dtype=v.dtype).scatter_add_(
            0, idx[1], v)
        dr = dr.clamp(min=1e-12).pow(-self.alpha)
        dc = dc.clamp(min=1e-12).pow(self.alpha - 1.0)
        return torch.sparse_coo_tensor(
            idx, v * dr[idx[0]] * dc[idx[1]], S.size()).coalesce()

    def _ncer_sparse(self, S):
        """NCER on existing edges only: w_ij <- Jaccard(i,j)^p.

        Exact equivalent of the previous dense blocked version, but the
        intersection counts are obtained with one sparse x sparse product
        (O(nnz*k) instead of O(N^3)) and no N x N dense tensor is built.
        """
        S = S.coalesce()
        n = S.size(0)
        idx = S.indices().cpu().numpy()
        Sc = sp.csr_matrix(
            (np.ones(idx.shape[1], dtype=np.float32), (idx[0], idx[1])),
            shape=(n, n))
        Sc.sum_duplicates()
        Sc.sort_indices()
        Inter = (Sc @ Sc.T).tocsr()                  # |N(i) & N(j)|
        W = Inter.multiply(Sc).tocsr()               # non-zero counts only
        W.sort_indices()
        del Inter
        Wc = W.tocoo()
        w_lin = Wc.row.astype(np.int64) * n + Wc.col  # row-major sorted
        Sc_c = Sc.tocoo()                             # row-major sorted
        s_lin = Sc_c.row.astype(np.int64) * n + Sc_c.col
        pos = np.searchsorted(w_lin, s_lin)
        posc = np.clip(pos, 0, max(len(w_lin) - 1, 0))
        found = (w_lin[posc] == s_lin) if len(w_lin) else np.zeros(
            len(s_lin), dtype=bool)
        inter = np.zeros(len(s_lin), dtype=np.float32)
        inter[found] = Wc.data[posc[found]]
        deg = np.asarray(Sc.sum(axis=1)).ravel()      # out-degree
        dr = deg[Sc_c.row]
        dc = deg[Sc_c.col]
        jac = (inter + 1e-6) / (dr + dc - inter + 1e-6)
        vals = np.power(jac, self.ncer_pow).astype(np.float32)
        ii = torch.from_numpy(np.stack([Sc_c.row, Sc_c.col])).to(S.device).long()
        vv = torch.from_numpy(vals).to(S.device)
        return torch.sparse_coo_tensor(ii, vv, (n, n)).coalesce()

    def _ncer_modality_purify(self, S_binary, block=2048):
        """Legacy dense NCER (kept for reference / debugging)."""
        S_binary = S_binary.float()
        n = S_binary.size(0)
        deg = S_binary.sum(dim=1, keepdim=True)          # (n, 1)

        out = torch.empty_like(S_binary)
        for i0 in range(0, n, block):
            i1 = min(i0 + block, n)
            blk = S_binary[i0:i1]                        # (bs, n)
            Inter = blk @ S_binary.T                     # (bs, n)，只占 bs*n
            Union = deg[i0:i1] + deg.T - Inter
            Jaccard = (Inter + 1e-6) / (Union + 1e-6)
            out[i0:i1] = blk * torch.pow(Jaccard, self.ncer_pow)
        return out

    def _hadamard_power(self, X, s):
        X = np.asarray(X, dtype=np.float32)
        return np.power(X, s)

    def _extremal_eigs(self, T, iters=25, tol=1e-8):
        """(lmin, lmax) of symmetric T by shifted power iteration."""
        n, dev, dt = T.size(0), T.device, T.dtype
        torch.manual_seed(42)
        v = torch.randn(n, 1, device=dev, dtype=dt)
        v = v / (v.norm() + 1e-12)
        lmax = 0.0
        for _ in range(iters):
            Av = T @ v
            if Av.norm() < 1e-12:
                break
            v = Av / Av.norm()
            l = float((v.T @ (T @ v)).item())
            if abs(l - lmax) < tol:
                lmax = l
                break
            lmax = l
        v = torch.randn(n, 1, device=dev, dtype=dt)
        v = v / (v.norm() + 1e-12)
        mu = 0.0
        for _ in range(iters):
            Bv = lmax * v - T @ v
            if Bv.norm() < 1e-12:
                break
            v = Bv / Bv.norm()
            m = float((v.T @ (lmax * v - T @ v)).item())
            if abs(m - mu) < tol:
                mu = m
                break
            mu = m
        return lmax - mu, lmax

    def _poly_filter(self, Pbar, name, coeffs):
        """Polynomial graph filter with lmin/lmax spectrum normalization."""
        T = torch.from_numpy(np.asarray(Pbar, dtype=np.float32)).to(self.device)
        lmin, lmax = self._extremal_eigs(T)
        lam = lmax - lmin
        if lam <= 0:
            lam = 1.0
        print('[MM-GF Fast] %s graph lmin=%.6f lmax=%.6f lambda*=%.6f'
              % (name, lmin, lmax, lam), flush=True)
        T.diagonal().sub_(lmin)
        T /= lam
        acc = None
        Tk = T
        for k, c in enumerate(coeffs, 1):
            if k > 1:
                Tk = Tk @ T
            acc = c * Tk if acc is None else acc + c * Tk
        return acc

    def calculate_loss(self, interaction):
        # training-free model: no learnable parameters, zero loss.
        return torch.tensor(0.0)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        return self.scores_matrix[user]