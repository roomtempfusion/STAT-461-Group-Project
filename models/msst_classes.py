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
        t0 = idx
        t_in_end = t0 + self.input_len
        t_out_end = t_in_end + self.horizon

        # Inputs
        X_state_seq = self.X_state[t0:t_in_end]      # (T_in, S, F_macro)
        X_county_seq = self.X_county[t0:t_in_end]    # (T_in, M, F_micro)

        # Future targets from state data
        # (horizon, S, F_macro)
        y_window = self.X_state[t_in_end:t_out_end]

        # Only variant dims: (horizon, S, v_out)
        y_window = y_window[:, :, self.target_variant_indices]

        # Model predicts (S, horizon, v_out), so transpose
        y_true = np.transpose(y_window, (1, 0, 2))

        return (
            torch.from_numpy(X_state_seq),    # (T_in, S, F_macro)
            torch.from_numpy(X_county_seq),   # (T_in, M, F_micro)
            torch.from_numpy(y_true),         # (S, horizon, v_out)
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
