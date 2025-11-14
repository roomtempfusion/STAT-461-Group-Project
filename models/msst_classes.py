import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset
import numpy as np

class MacroGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, A_state):
        super().__init__()
        # Convert adjacency matrix -> edge_index / edge_weight
        A = torch.tensor(A_state, dtype=torch.float)
        edge_index, edge_weight = dense_to_sparse(A)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):  # x: (S, F)
        h = self.act(self.conv1(x, self.edge_index, self.edge_weight))
        h = self.drop(h)
        h = self.conv2(h, self.edge_index, self.edge_weight)
        return h  # (S, out_dim)


class MicroGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, A_county_global, state_of_county):
        super().__init__()
        A = torch.tensor(A_county_global, dtype=torch.float)
        edge_index, edge_weight = dense_to_sparse(A)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_weight', edge_weight)

        self.state_of_county = torch.tensor(state_of_county, dtype=torch.long)
        self.num_states = int(self.state_of_county.max().item() + 1)

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):  # x: (M, F_micro)
        h = self.act(self.conv1(x, self.edge_index, self.edge_weight))
        h = self.drop(h)
        h = self.conv2(h, self.edge_index, self.edge_weight)  # (M, out_dim)

        # Pool per state: mean pooling
        S = self.num_states
        out = torch.zeros(S, h.size(1), device=h.device)
        counts = torch.zeros(S, 1, device=h.device)

        # scatter-add
        out.index_add_(0, self.state_of_county, h)
        counts.index_add_(0, self.state_of_county,
                          torch.ones_like(counts[self.state_of_county]))
        out = out / counts.clamp(min=1.0)

        return out  # (S, out_dim)


class MSSTVariant(nn.Module):
    def __init__(self, macro_in, micro_in, hidden_gcn, hidden_gru, horizon, v_out,
                 A_state, A_county_global, state_of_county):
        super().__init__()
        self.macro_gcn = MacroGCN(macro_in, hidden_gcn, hidden_gcn, A_state)
        self.micro_gcn = MicroGCN(micro_in, hidden_gcn, hidden_gcn,
                                  A_county_global, state_of_county)

        # learnable fusion scalars
        self.w_macro = nn.Parameter(torch.tensor(0.5))
        self.w_micro = nn.Parameter(torch.tensor(0.5))

        self.gru = nn.GRU(input_size=hidden_gcn, hidden_size=hidden_gru,
                          batch_first=False)  # (T, B=N_states, d_h)

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_gru, hidden_gru),
            nn.ReLU(),
            nn.Linear(hidden_gru, horizon * v_out)
        )
        self.horizon = horizon
        self.v_out = v_out

    def forward(self, X_state_seq, X_county_seq):
        """
        X_state_seq:  (T, S, F_macro)
        X_county_seq: (T, M, F_micro)
        returns:      (S, horizon, v_out)
        """
        T, S, _ = X_state_seq.shape
        fused_list = []
        for t in range(T):
            macro_emb = self.macro_gcn(X_state_seq[t])    # (S, d_g)
            # micro_emb = self.micro_gcn(X_county_seq[t])   # (S, d_g)

            # fused = self.w_macro * macro_emb + self.w_micro * micro_emb  # (S, d_g)
            # fused_list.append(fused)
            fused_list.append(macro_emb)

        H_fused = torch.stack(fused_list, dim=0)  # (T, S, d_g)

        # GRU expects (T, batch, features)
        H_gru, _ = self.gru(H_fused)  # (T, S, d_gru)
        H_last = H_gru[-1]            # (S, d_gru)

        out = self.pred_head(H_last)  # (S, horizon * v_out)
        out = out.view(S, self.horizon, self.v_out)
        return out


class EpidemicDataset(Dataset):
    def __init__(self, X_state, X_county, input_len, horizon, target_variant_indices):
        """
        X_state:  (T, S, F_macro)
        X_county: (T, M, F_micro)
        """
        assert X_state.shape[0] == X_county.shape[0]
        self.X_state = X_state.astype(np.float32)
        self.X_county = X_county.astype(np.float32)
        self.input_len = input_len
        self.horizon = horizon

        self.T, self.S, self.F_macro = X_state.shape
        self.M = X_county.shape[1]

        self.target_variant_indices = np.array(target_variant_indices, dtype=int)
        self.v_out = len(self.target_variant_indices)

        self.num_samples = self.T - self.input_len - self.horizon + 1
        assert self.num_samples > 0, "Not enough time steps!"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1) Inputs
        X_state_seq = self.X_state[idx: idx + self.input_len]  # (T_in, S, F)
        X_county_seq = self.X_county[idx: idx + self.input_len]  # (T_in, M, F_micro)

        # 2) Raw future targets
        t_start = idx + self.input_len
        t_end = t_start + self.horizon  # not inclusive
        Y_future = self.X_state[t_start:t_end, :, :]  # (H, S, F)
        Y_future = Y_future[:, :, self.target_variant_indices]  # (H, S, V)
        y_raw = np.transpose(Y_future, (1, 0, 2))  # (S, H, V)

        # 3) Baseline (e.g., 7-day MA of last K days in input)
        # Use last K days of the input window
        K = min(7, self.input_len)
        last_k = X_state_seq[-K:, :, :]  # (K, S, F)
        last_k_cases = last_k[:, :, self.target_variant_indices]  # (K, S, V)
        baseline = last_k_cases.mean(axis=0)  # (S, V)
        baseline_full = np.repeat(baseline[:, None, :], self.horizon, axis=1)  # (S, H, V)

        # 4) Residual target
        y_resid = y_raw - baseline_full

        return (
            torch.from_numpy(X_state_seq).float(),
            torch.from_numpy(X_county_seq).float(),
            torch.from_numpy(y_resid).float(),
            torch.from_numpy(baseline_full).float(),  # needed at eval to reconstruct
        )


class GRUOnly(nn.Module):
    def __init__(self, F_macro, hidden_gru, horizon, v_out):
        super().__init__()
        self.gru = nn.GRU(input_size=F_macro, hidden_size=hidden_gru, batch_first=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_gru, hidden_gru),
            nn.ReLU(),
            nn.Linear(hidden_gru, horizon * v_out),
        )
        self.horizon = horizon
        self.v_out = v_out

    def forward(self, X_state_seq, X_county_seq=None):
        H, _ = self.gru(X_state_seq)       # (T_in, S, hidden_gru)
        H_last = H[-1]                     # (S, hidden_gru)
        out = self.head(H_last)            # (S, horizon*v_out)
        return out.view(H_last.size(0), self.horizon, self.v_out)


class MacroGCNGRUResidualSkip(nn.Module):
    def __init__(self, macro_gcn, in_dim, gcn_out_dim, hidden_gru, horizon, v_out):
        super().__init__()
        self.macro_gcn = macro_gcn
        self.in_dim = in_dim
        self.gcn_out_dim = gcn_out_dim

        # GRU sees [raw features, GCN emb]
        self.gru = nn.GRU(
            input_size=in_dim + gcn_out_dim,
            hidden_size=hidden_gru,
            batch_first=False  # (T, S, d)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_gru, hidden_gru),
            nn.ReLU(),
            nn.Linear(hidden_gru, horizon * v_out),
        )

        self.horizon = horizon
        self.v_out = v_out

    def forward(self, X_state_seq, X_county_seq=None):
        # X_state_seq: (T_in, S, F_macro)
        T_in, S, F_macro = X_state_seq.shape

        fused_list = []
        for t in range(T_in):
            x_t = X_state_seq[t]             # (S, F_macro)
            g_t = self.macro_gcn(x_t)       # (S, gcn_out_dim)
            fused_t = torch.cat([x_t, g_t], dim=-1)  # (S, F_macro + gcn_out_dim)
            fused_list.append(fused_t)

        H_seq = torch.stack(fused_list, dim=0)      # (T_in, S, F_macro + gcn_out_dim)
        H_out, _ = self.gru(H_seq)                  # (T_in, S, hidden_gru)
        H_last = H_out[-1]                          # (S, hidden_gru)

        out = self.head(H_last)                     # (S, horizon*v_out)
        return out.view(S, self.horizon, self.v_out)


class MacroMicroGCNGRUResidual(nn.Module):
    def __init__(
        self,
        macro_gcn,         # instance of MacroGCN
        micro_gcn,         # instance of MicroGCN (your class)
        in_dim_state,      # F_macro: #variant features per state
        macro_out_dim,     # out_dim of MacroGCN
        micro_out_dim,     # out_dim of MicroGCN
        hidden_gru,
        horizon,
        v_out,
    ):
        super().__init__()
        self.macro_gcn = macro_gcn
        self.micro_gcn = micro_gcn

        self.in_dim_state = in_dim_state
        self.macro_out_dim = macro_out_dim
        self.micro_out_dim = micro_out_dim

        # GRU sees: [raw state features, macro emb, micro emb]
        gru_input_dim = in_dim_state + macro_out_dim + micro_out_dim

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_gru,
            batch_first=False  # (T, S, d)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_gru, hidden_gru),
            nn.ReLU(),
            nn.Linear(hidden_gru, horizon * v_out),
        )

        self.horizon = horizon
        self.v_out = v_out

    def forward(self, X_state_seq, X_county_seq):
        """
        X_state_seq:  (T_in, S, F_macro)
        X_county_seq: (T_in, M, F_micro)
          - MicroGCN.forward expects x_county_t: (M, F_micro)
          - and returns state-level pooled: (S, micro_out_dim)
        """
        T_in, S, F_macro = X_state_seq.shape
        _, M, F_micro = X_county_seq.shape

        fused_list = []

        for t in range(T_in):
            # State block at time t
            x_state_t = X_state_seq[t]                # (S, F_macro)
            h_macro_t = self.macro_gcn(x_state_t)     # (S, macro_out_dim)

            # County block at time t
            x_county_t = X_county_seq[t]              # (M, F_micro)
            h_micro_t = self.micro_gcn(x_county_t)    # (S, micro_out_dim) pooled to states

            # Sanity: ensure MicroGCN returns same #states
            # (optional, but nice while debugging)
            # assert h_micro_t.shape[0] == S

            # Fuse raw + macro + micro
            h_fused_t = torch.cat(
                [x_state_t, h_macro_t, h_micro_t],
                dim=-1
            )  # (S, F_macro + macro_out_dim + micro_out_dim)

            fused_list.append(h_fused_t)

        H_seq = torch.stack(fused_list, dim=0)        # (T_in, S, gru_input_dim)
        H_out, _ = self.gru(H_seq)                    # (T_in, S, hidden_gru)
        H_last = H_out[-1]                            # (S, hidden_gru)

        out = self.head(H_last)                       # (S, horizon * v_out)
        return out.view(S, self.horizon, self.v_out)  # residuals (S, H, V)
