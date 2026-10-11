import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import glob
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1.0
    def transform(self, X):
        return (X - self.mean) / self.std


def stratified_split(indices, labels, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    train_list = []
    test_list = []
    for c in np.unique(labels[indices]):
        cls_idx = indices[labels[indices] == c]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * test_ratio))
        test_list.append(cls_idx[:n_test])
        train_list.append(cls_idx[n_test:])
    train_idx = np.concatenate(train_list, axis=0)
    test_idx = np.concatenate(test_list, axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def accuracy_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float((y_true == y_pred).mean())


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report_np(y_true, y_pred, num_classes, label_names=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    report_lines = []
    report_lines.append("class\tprecision\trecall\tf1-score\tsupport")
    eps = 1e-12
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        name = str(c) if label_names is None else label_names.get(c, str(c))
        report_lines.append(f"{name}\t{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}\t\t{support}")
    acc = accuracy_np(y_true, y_pred)
    report_lines.append(f"\naccuracy\t{acc:.4f}")
    return "\n".join(report_lines)


class PlanetaryGearDatasetV32(Dataset):
    def __init__(self, root_dir, num_imfs=6, signal_len=3072, augment=False,
                 nrows=50, use_ceemdan=True, ceemdan_trials=50, ceemdan_noise_width=0.2):
        self.root_dir = root_dir
        self.J = num_imfs
        self.L = signal_len
        self.augment = augment
        self.nrows = nrows
        self.label_map = {'内外圈': 0, '内滚': 1, '外滚': 2, '正常': 3}
        self.label_names = {0: '内外圈', 1: '内滚', 2: '外滚', 3: '正常'}
        self.vibration_samples = None
        self.current_samples = None
        self.labels = None
        self.use_ceemdan = use_ceemdan
        self._ceemdan = None
        from PyEMD import CEEMDAN
        self._ceemdan = CEEMDAN(trials=ceemdan_trials)
        self._ceemdan.noise_width = ceemdan_noise_width
        self._load_data()
    def _load_data(self):
        fault_types = ['内外圈', '内滚', '外滚', '正常']
        loads = ['0', '2', '4', '6']
        speeds = ['300', '600', '900']
        all_vib = []
        all_cur = []
        all_lbl = []
        for fault_type in fault_types:
            for load in loads:
                for speed in speeds:
                    vib_pattern = os.path.join(self.root_dir, fault_type, '振动', load, f"*转速{speed}*.csv")
                    vib_files = glob.glob(vib_pattern)
                    cur_pattern = os.path.join(self.root_dir, fault_type, '电流', load, '1', f"*转速{speed}*.csv")
                    cur_files = glob.glob(cur_pattern)
                    if vib_files and cur_files:
                        try:
                            vib_df = pd.read_csv(vib_files[0], skiprows=1, header=None, nrows=self.nrows)
                            cur_df = pd.read_csv(cur_files[0], skiprows=1, header=None, nrows=self.nrows)
                            min_rows = min(vib_df.shape[0], cur_df.shape[0])
                            for i in range(min_rows):
                                v_row = vib_df.iloc[i, :self.L].values
                                if len(v_row) < self.L:
                                    continue
                                c1 = cur_df.iloc[i, :self.L].values
                                if len(c1) < self.L:
                                    continue
                                all_vib.append(v_row.astype(np.float32))
                                all_cur.append(c1.astype(np.float32))
                                all_lbl.append(self.label_map[fault_type])
                        except Exception:
                            pass
        all_vib = np.array(all_vib, dtype=np.float32)
        all_cur = np.array(all_cur, dtype=np.float32)
        all_lbl = np.array(all_lbl, dtype=np.int64)
        if all_vib.shape[0] == 0:
            raise RuntimeError("数据读入为 0：请确认 root_dir、文件夹名、文件名包含“转速xxx”。")
        self.vibration_samples = all_vib
        self.current_samples = all_cur
        self.labels = all_lbl
    def _ceemdan_decomposition(self, signal):
        x = np.asarray(signal, dtype=np.float32)
        imfs = self._ceemdan.ceemdan(x)
        imfs = np.asarray(imfs, dtype=np.float32)
        if imfs.ndim != 2:
            imfs = np.atleast_2d(imfs)
        K = imfs.shape[0]
        if K >= self.J:
            imfs = imfs[:self.J]
        else:
            pad = np.zeros((self.J - K, self.L), dtype=np.float32)
            imfs = np.concatenate([imfs, pad], axis=0)
        return imfs
    def __len__(self):
        return int(len(self.labels))
    def __getitem__(self, idx):
        vib_raw = self.vibration_samples[idx].copy()
        cur_raw = self.current_samples[idx].copy()
        y = int(self.labels[idx])
        if self.augment and np.random.rand() < 0.5:
            vib_raw = vib_raw + np.random.normal(0, 0.01, size=vib_raw.shape).astype(np.float32)
            cur_raw = cur_raw + np.random.normal(0, 0.01, size=cur_raw.shape).astype(np.float32)
        vib_imfs = self._ceemdan_decomposition(vib_raw)
        cur_imfs = self._ceemdan_decomposition(cur_raw)
        return torch.from_numpy(vib_imfs), torch.from_numpy(cur_imfs), torch.tensor(y, dtype=torch.long)


class StateAwareNetwork(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.W1 = nn.Linear(2, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = torch.tanh(self.W1(x))
        s = self.W2(h)
        return torch.softmax(s, dim=1)


class DeepStreamCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, out_dim)
    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return self.fc(z)


class EvidentialHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, feat):
        logits = self.fc(feat)
        e = F.softplus(logits)
        alpha = e + 1.0
        return e, alpha


class DSEvidentialFusion(nn.Module):
    def __init__(self, num_classes: int, gamma=5.0, lam=0.5):
        super().__init__()
        self.K = num_classes
        self.gamma = gamma
        self.lam = lam
    def alpha_to_bu(self, alpha: torch.Tensor):
        S = torch.sum(alpha, dim=1, keepdim=True)
        e = alpha - 1.0
        b = e / (S + 1e-8)
        u = self.K / (S + 1e-8)
        return b, u
    def bu_to_alpha(self, b: torch.Tensor, u: torch.Tensor):
        S = self.K / (u + 1e-8)
        e = b * S
        return e + 1.0
    def forward(self, alpha_v: torch.Tensor, alpha_c: torch.Tensor, return_info: bool = False):
        b_v, u_v = self.alpha_to_bu(alpha_v)
        b_c, u_c = self.alpha_to_bu(alpha_c)
        score_v = self.gamma * ((1.0 - u_v) + self.lam * u_c)
        score_c = self.gamma * ((1.0 - u_c) + self.lam * u_v)
        eta = torch.cat([score_v, score_c], dim=1)
        eta = torch.softmax(eta, dim=1)
        eta_v = eta[:, 0:1]
        eta_c = eta[:, 1:2]
        b_v = b_v * eta_v
        b_c = b_c * eta_c
        u_v2 = torch.clamp(1.0 - b_v.sum(dim=1, keepdim=True), 1e-8, 1.0)
        u_c2 = torch.clamp(1.0 - b_c.sum(dim=1, keepdim=True), 1e-8, 1.0)
        bb = torch.bmm(b_v.unsqueeze(2), b_c.unsqueeze(1))
        conflict = bb.sum(dim=(1, 2)) - torch.sum(b_v * b_c, dim=1)
        denom = (1.0 - conflict).unsqueeze(1) + 1e-8
        b_f = (b_v * b_c + b_v * u_c2 + b_c * u_v2) / denom
        u_f = (u_v2 * u_c2) / denom
        alpha_f = self.bu_to_alpha(b_f, u_f)
        if not return_info:
            return alpha_f
        conf_v = 1.0 - u_v
        conf_c = 1.0 - u_c
        w_sum = conf_v + conf_c + 1e-8
        w_v = conf_v / w_sum
        w_c = conf_c / w_sum
        info = {
            "u_v": u_v.detach(), "u_c": u_c.detach(),
            "w_v": eta_v.detach(), "w_c": eta_c.detach(),
            "conflict": conflict.detach(),
        }
        return alpha_f, info


def dirichlet_kl_to_uniform(alpha, num_classes, eps=1e-8):
    K = num_classes
    alpha0 = torch.sum(alpha, dim=1, keepdim=True)
    t1 = torch.lgamma(alpha0 + eps) - torch.sum(torch.lgamma(alpha + eps), dim=1, keepdim=True)
    t2 = -torch.lgamma(torch.tensor(float(K), device=alpha.device) + eps)
    digamma_alpha = torch.digamma(alpha + eps)
    digamma_sum = torch.digamma(alpha0 + eps)
    t3 = torch.sum((alpha - 1.0) * (digamma_alpha - digamma_sum), dim=1, keepdim=True)
    kl = t1 + t2 + t3
    return kl.squeeze(1)


def edl_mse_risk(alpha, y_onehot):
    S = torch.sum(alpha, dim=1, keepdim=True)
    p = alpha / (S + 1e-8)
    mse = torch.sum((y_onehot - p) ** 2, dim=1)
    var = torch.sum(p * (1 - p) / (S + 1.0), dim=1)
    return mse + var


def focal_loss_from_probs(probs, y, gamma=2.0, alpha=None, eps=1e-8):
    if probs.dim() != 2:
        raise ValueError(f"probs must be 2D (B,K), got {probs.shape}")
    B, K = probs.shape
    pt = probs.gather(1, y.view(-1, 1)).clamp_min(eps).squeeze(1)
    log_pt = torch.log(pt)
    focal = (1.0 - pt) ** gamma
    if alpha is None:
        loss = -focal * log_pt
    else:
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, device=probs.device, dtype=probs.dtype)
        a_t = alpha.gather(0, y).to(probs.dtype)
        loss = -a_t * focal * log_pt
    return loss.mean()



def edl_loss(alpha, y, num_classes, epoch, annealing_steps=10):
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    risk = edl_mse_risk(alpha, y_onehot)
    anneal = min(1.0, float(epoch) / float(annealing_steps))
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = dirichlet_kl_to_uniform(alpha_tilde, num_classes)
    return torch.mean(risk + anneal * kl)


class V32_Model(nn.Module):
    def __init__(self, num_classes=4, num_imfs=6, signal_len=3072,
                 deep_feat_dim=128, sa_hidden=32,
                 lambda_phys=0.1, lambda_sparse=0.01):
        super().__init__()
        self.K = num_classes
        self.J = num_imfs
        self.L = signal_len
        self.lambda_phys = lambda_phys
        self.lambda_sparse = lambda_sparse
        self.sa_vib = StateAwareNetwork(hidden_dim=sa_hidden)
        self.sa_cur = StateAwareNetwork(hidden_dim=sa_hidden)
        self.cnn_vib = DeepStreamCNN(out_dim=deep_feat_dim)
        self.cnn_cur = DeepStreamCNN(out_dim=deep_feat_dim)
        self.head_vib = EvidentialHead(in_dim=deep_feat_dim, num_classes=num_classes)
        self.head_cur = EvidentialHead(in_dim=deep_feat_dim, num_classes=num_classes)
        self.ds = DSEvidentialFusion(num_classes=num_classes, gamma=5.0, lam=0.5)
        self.register_buffer("vib_ref_mu", torch.zeros(num_imfs))
        self.register_buffer("vib_ref_std", torch.ones(num_imfs))
        self.register_buffer("vib_ref_kurt", torch.ones(num_imfs) * 3.0)
        self.register_buffer("cur_ref_mu", torch.zeros(num_imfs))
        self.register_buffer("cur_ref_std", torch.ones(num_imfs))
        self.register_buffer("cur_ref_kurt", torch.ones(num_imfs) * 3.0)
    def load_reference(self, ref: dict):
        self.vib_ref_mu.copy_(ref["vib_mu"])
        self.vib_ref_std.copy_(ref["vib_std"])
        self.vib_ref_kurt.copy_(ref["vib_kurt"])
        self.cur_ref_mu.copy_(ref["cur_mu"])
        self.cur_ref_std.copy_(ref["cur_std"])
        self.cur_ref_kurt.copy_(ref["cur_kurt"])
    def _physical_e_and_P(self, imfs, ref_mu, ref_std, ref_kurt):
        mu = imfs.mean(dim=2)
        std = imfs.std(dim=2) + 1e-8
        xc = imfs - mu.unsqueeze(-1)
        var = torch.mean(xc ** 2, dim=2) + 1e-8
        m4 = torch.mean(xc ** 4, dim=2)
        kurt = m4 / (var ** 2)
        mu_ref = ref_mu.unsqueeze(0)
        std_ref = ref_std.unsqueeze(0)
        term1 = torch.log((std_ref + 1e-8) / (std + 1e-8))
        term2 = (std ** 2 + (mu - mu_ref) ** 2) / (2.0 * (std_ref ** 2 + 1e-8))
        d_kl = term1 + term2 - 0.5
        d_kurt = torch.abs(kurt - ref_kurt.unsqueeze(0))
        e = torch.stack([d_kl, d_kurt], dim=2)
        P = torch.sqrt(d_kl ** 2 + d_kurt ** 2 + 1e-8)
        return e, P
    def _deep_features(self, imfs, cnn):
        mu = imfs.mean(dim=2, keepdim=True)
        std = imfs.std(dim=2, keepdim=True) + 1e-8
        z = (imfs - mu) / std
        B, J, L = z.shape
        z = z.reshape(B * J, 1, L)
        f = cnn(z)
        D = f.shape[1]
        return f.reshape(B, J, D)
    @staticmethod
    def _aggregate(feats, alpha):
        return torch.sum(feats * alpha, dim=1)
    @staticmethod
    def physical_consistency_loss(alpha, P, eps=1e-8):
        a = alpha.squeeze(-1)
        num = torch.sum(a * P, dim=1)
        den = torch.sqrt(torch.sum(a * a, dim=1) + eps) * torch.sqrt(torch.sum(P * P, dim=1) + eps)
        cos = num / (den + eps)
        return torch.mean(1.0 - cos)
    @staticmethod
    def sparse_loss(alpha, eps=1e-8):
        a = torch.clamp(alpha.squeeze(-1), min=eps, max=1.0)
        return torch.mean(-torch.sum(a * torch.log(a + eps), dim=1))
    def forward(self, vib_imfs, cur_imfs):
        vib_e, vib_P = self._physical_e_and_P(vib_imfs, self.vib_ref_mu, self.vib_ref_std, self.vib_ref_kurt)
        cur_e, cur_P = self._physical_e_and_P(cur_imfs, self.cur_ref_mu, self.cur_ref_std, self.cur_ref_kurt)
        alpha_v = self.sa_vib(vib_e)
        alpha_c = self.sa_cur(cur_e)
        vib_deep = self._deep_features(vib_imfs, self.cnn_vib)
        cur_deep = self._deep_features(cur_imfs, self.cnn_cur)
        vib_feat = self._aggregate(vib_deep, alpha_v)
        cur_feat = self._aggregate(cur_deep, alpha_c)
        _, alpha_dir_v = self.head_vib(vib_feat)
        _, alpha_dir_c = self.head_cur(cur_feat)
        alpha_f, ds_info = self.ds(alpha_dir_v, alpha_dir_c, return_info=True)
        aux = {
            "alpha_v": alpha_v, "alpha_c": alpha_c,
            "vib_P": vib_P, "cur_P": cur_P,
            "ds": ds_info,
        }
        return alpha_dir_v, alpha_dir_c, alpha_f, aux
@torch.no_grad()


def build_reference_stats_from_dataset(dataset, train_indices, normal_label=3, device="cpu"):
    vib_mu_acc, vib_std_acc, vib_kurt_acc = [], [], []
    cur_mu_acc, cur_std_acc, cur_kurt_acc = [], [], []
    n_norm = 0
    for idx in train_indices:
        vib_imfs, cur_imfs, y = dataset[idx]
        if int(y.item()) != normal_label:
            continue
        vib_imfs = vib_imfs.to(device)
        cur_imfs = cur_imfs.to(device)
        v_mu = vib_imfs.mean(dim=1)
        v_std = vib_imfs.std(dim=1) + 1e-8
        v_xc = vib_imfs - v_mu.unsqueeze(1)
        v_var = torch.mean(v_xc ** 2, dim=1) + 1e-8
        v_m4 = torch.mean(v_xc ** 4, dim=1)
        v_kurt = v_m4 / (v_var ** 2)
        c_mu = cur_imfs.mean(dim=1)
        c_std = cur_imfs.std(dim=1) + 1e-8
        c_xc = cur_imfs - c_mu.unsqueeze(1)
        c_var = torch.mean(c_xc ** 2, dim=1) + 1e-8
        c_m4 = torch.mean(c_xc ** 4, dim=1)
        c_kurt = c_m4 / (c_var ** 2)
        vib_mu_acc.append(v_mu)
        vib_std_acc.append(v_std)
        vib_kurt_acc.append(v_kurt)
        cur_mu_acc.append(c_mu)
        cur_std_acc.append(c_std)
        cur_kurt_acc.append(c_kurt)
        n_norm += 1
    if n_norm == 0:
        raise RuntimeError("训练集中没有正常样本（label=3），无法构建参考参数。")
    vib_mu = torch.stack(vib_mu_acc, dim=0).mean(dim=0)
    vib_std = torch.stack(vib_std_acc, dim=0).mean(dim=0)
    vib_kurt = torch.stack(vib_kurt_acc, dim=0).mean(dim=0)
    cur_mu = torch.stack(cur_mu_acc, dim=0).mean(dim=0)
    cur_std = torch.stack(cur_std_acc, dim=0).mean(dim=0)
    cur_kurt = torch.stack(cur_kurt_acc, dim=0).mean(dim=0)
    return {
        "vib_mu": vib_mu.detach(),
        "vib_std": vib_std.detach(),
        "vib_kurt": vib_kurt.detach(),
        "cur_mu": cur_mu.detach(),
        "cur_std": cur_std.detach(),
        "cur_kurt": cur_kurt.detach(),
        "n_norm": n_norm
    }


def train_one_epoch(model, loader, optimizer, device, epoch, annealing_steps=10):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    
    J = model.J
    sum_imf_v = torch.zeros(J, device=device)
    sum_imf_c = torch.zeros(J, device=device)
    nb = 0
    for vib_imfs, cur_imfs, y in loader:
        vib_imfs = vib_imfs.to(device)
        cur_imfs = cur_imfs.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        alpha_v, alpha_c, alpha_f, aux = model(vib_imfs, cur_imfs)
        loss_f = edl_loss(alpha_f, y, model.K, epoch, annealing_steps=annealing_steps)
        loss_cls = loss_f
        loss_phy = (model.physical_consistency_loss(aux["alpha_v"], aux["vib_P"]) +
                   model.physical_consistency_loss(aux["alpha_c"], aux["cur_P"]))
        loss_sparse = model.sparse_loss(aux["alpha_v"]) + model.sparse_loss(aux["alpha_c"])
        loss = loss_cls + model.lambda_phys * loss_phy + model.lambda_sparse * loss_sparse
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            p = alpha_f / (torch.sum(alpha_f, dim=1, keepdim=True) + 1e-8)
            pred = torch.argmax(p, dim=1)
            acc = (pred == y).float().mean().item()
            av = aux["alpha_v"].squeeze(-1)
            ac = aux["alpha_c"].squeeze(-1)
            sum_imf_v += av.mean(dim=0)
            sum_imf_c += ac.mean(dim=0)
            nb += 1
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs
    return total_loss / max(n, 1), total_acc / max(n, 1)
@torch.no_grad()


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for vib_imfs, cur_imfs, y in loader:
        vib_imfs = vib_imfs.to(device)
        cur_imfs = cur_imfs.to(device)
        _, _, alpha_f, _ = model(vib_imfs, cur_imfs)
        p = alpha_f / (torch.sum(alpha_f, dim=1, keepdim=True) + 1e-8)
        pred = torch.argmax(p, dim=1).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    ys = np.concatenate(ys, axis=0).reshape(-1)
    preds = np.concatenate(preds, axis=0).reshape(-1)
    return accuracy_np(ys, preds), ys, preds


def main():
    set_seed(42)
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = r"C:****\U****sers\30819\Deskto&#x70;****\迁****移学&#x4E60;****\信****息融&#x5408;****\数****据+代&#x7801;****\数****据"
    num_imfs = 6
    signal_len = 3072
    batch_size = 32
    lr = 1e-3
    epochs = 10
    dataset = PlanetaryGearDatasetV32(
        root_dir=data_root,
        num_imfs=num_imfs,
        signal_len=signal_len,
        augment=True,
        nrows=100,
        use_ceemdan=True
    )
    idx_all = np.arange(len(dataset))
    train_idx, test_idx = stratified_split(idx_all, dataset.labels, test_ratio=0.2, seed=42)
    train_idx, val_idx = stratified_split(train_idx, dataset.labels, test_ratio=0.2, seed=43)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model = V32_Model(
        num_classes=4,
        num_imfs=num_imfs,
        signal_len=signal_len,
        deep_feat_dim=128,
        sa_hidden=32,
        lambda_phys=0.1,
        lambda_sparse=0.01
    ).to(device)
    ref = build_reference_stats_from_dataset(dataset, train_idx, normal_label=3, device=device)
    model.load_reference(ref)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = -1
    best_state = None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, epoch=ep)
        val_acc, _, _ = evaluate(model, val_loader, device)
        test_acc, _, _ = evaluate(model, test_loader, device)
        dt = time.time() - t0
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
if __name__ == "__main__":
    main()
