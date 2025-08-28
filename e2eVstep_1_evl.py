#!/usr/bin/env python3
# e2eVstep_1_eval.py developed by Dr Manish Ghatge, Scientist, IITM Pune
# Full working version: Conv→ViT→UNet, memmap dataset, scaler, multi-GPU compatible

import os, json, argparse
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def load_meta(meta_path: str):
    with open(meta_path,'r') as f:
        meta = json.load(f)
    shape = tuple(meta['shape'])
    dtype = np.dtype(meta.get('dtype','float32'))
    return shape, dtype

def open_memmap(npy_path: str, meta_path: str):
    shape, dtype = load_meta(meta_path)
    return np.memmap(npy_path, dtype=dtype, mode='r', shape=shape)

def slice_var(memmap_obj: np.memmap, t_idx: int, level_reduce='first', level_index=0):
    shape = memmap_obj.shape
    ndim = len(shape)

    def _t_slice(a, t):
        if a.shape[0] > t:
            return a[t]
        else:
            raise IndexError(f"time index {t} out of range (len {a.shape[0]})")

    if ndim == 1:
        return np.array(memmap_obj, dtype=np.float32).reshape(1,1)
    if ndim == 2:
        if shape[0]>50:
            return np.array(_t_slice(memmap_obj,t_idx), dtype=np.float32)
        else:
            return np.array(memmap_obj, dtype=np.float32)
    if ndim == 3:
        if shape[0]>50:
            return np.array(_t_slice(memmap_obj,t_idx), dtype=np.float32)
        else:
            return np.array(memmap_obj, dtype=np.float32).squeeze()
    if ndim == 4:
        if shape[0]>50:
            tslice = _t_slice(memmap_obj,t_idx)
            if tslice.ndim==3:
                if level_reduce=='first':
                    return np.array(tslice[level_index], dtype=np.float32)
                elif level_reduce=='mean':
                    return np.array(tslice.mean(axis=0), dtype=np.float32)
                else:
                    raise ValueError("level_reduce must be 'first' or 'mean'")
            else:
                return np.array(tslice, dtype=np.float32).squeeze()
        else:
            tslice = memmap_obj[0, t_idx]
            return np.array(tslice, dtype=np.float32)
    if ndim == 5:
        if shape[1]>50:
            arr = memmap_obj[:, t_idx]
            arr = arr[0]
            if arr.ndim==3:
                if level_reduce=='first':
                    return np.array(arr[level_index], dtype=np.float32)
                else:
                    return np.array(arr.mean(axis=0), dtype=np.float32)
            else:
                return np.array(arr, dtype=np.float32).squeeze()
        else:
            arr = memmap_obj[0, t_idx]
            if arr.ndim==3:
                if level_reduce=='first':
                    return np.array(arr[level_index], dtype=np.float32)
                else:
                    return np.array(arr.mean(axis=0), dtype=np.float32)
            else:
                return np.array(arr, dtype=np.float32).squeeze()
    raise ValueError(f"Unsupported memmap ndim={ndim}, shape={shape}")


class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted=False
    def fit(self,data:torch.Tensor):
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True)
        self.fitted=True
    def transform(self,data:torch.Tensor):
        if not self.fitted: raise RuntimeError("Scaler not fitted yet")
        return (data - self.mean)/(self.std+1e-8)
    def inverse_transform(self,data:torch.Tensor):
        if not self.fitted: raise RuntimeError("Scaler not fitted yet")
        return data*(self.std+1e-8)+self.mean



class WeatherDataset(Dataset):
    def __init__(self, data_dir: str, target_vars: List[str], input_vars: List[str] = None,
                 device='cuda', level_reduce='first', level_index=0):
        self.device = device
        self.target_vars = target_vars
        self.input_vars = input_vars if input_vars is not None else target_vars
        self.data = {}
        self.scalers = {}

        # ---- load all input + target variables (unchanged) ----
        for v in self.input_vars + self.target_vars:
            npy_path = os.path.join(data_dir, f"{v}.npy")
            meta_path = os.path.join(data_dir, f"{v}.meta.json")
            memmap_obj = open_memmap(npy_path, meta_path)
            arr_list = [slice_var(memmap_obj, t, level_reduce, level_index)
                        for t in range(memmap_obj.shape[0])]
            data_tensor = torch.tensor(np.stack(arr_list), dtype=torch.float32, device='cpu')  # (N,H,W)
            self.data[v] = data_tensor

            scaler = TorchStandardScaler()
            scaler.fit(data_tensor)
            self.scalers[v] = scaler
            self.data[v] = scaler.transform(data_tensor)

        
        n_samples = self.data[self.target_vars[0]].shape[0]
        H = self.data[self.target_vars[0]].shape[1]
        W = self.data[self.target_vars[0]].shape[2]

        t = torch.arange(n_samples, dtype=torch.float32)  # 0..N-1
        t_norm = t / n_samples                            # 0..~1

        
        sin_t = torch.sin(2 * np.pi * t_norm)
        cos_t = torch.cos(2 * np.pi * t_norm)

        
        sin_maps = sin_t.view(n_samples, 1, 1).expand(-1, H, W).contiguous()
        cos_maps = cos_t.view(n_samples, 1, 1).expand(-1, H, W).contiguous()

        
        self.data["time_sin"] = sin_maps
        self.data["time_cos"] = cos_maps

        
        sin_scaler = TorchStandardScaler(); sin_scaler.fit(self.data["time_sin"])
        cos_scaler = TorchStandardScaler(); cos_scaler.fit(self.data["time_cos"])
        self.scalers["time_sin"] = sin_scaler; self.scalers["time_cos"] = cos_scaler
        self.data["time_sin"] = sin_scaler.transform(self.data["time_sin"])
        self.data["time_cos"] = cos_scaler.transform(self.data["time_cos"])

        
        if "time_sin" not in self.input_vars:
            self.input_vars.append("time_sin")
        if "time_cos" not in self.input_vars:
            self.input_vars.append("time_cos")
        # ---------------------------------------------------------------------------

    def __len__(self):
        return self.data[self.target_vars[0]].shape[0]

    def __getitem__(self, idx):
        X = [self.data[v][idx] for v in self.input_vars]   
        Y = [self.data[v][idx] for v in self.target_vars]  
        return torch.stack(X), torch.stack(Y)              




class VarEncoder(nn.Module):
    def __init__(self, in_ch=1, emb_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, emb_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(emb_ch, emb_ch, 3, padding=1), nn.ReLU()
        )
    def forward(self,x):
        return self.net(x)

class SimplePatchViT(nn.Module):
    def __init__(self, emb_ch, patch_size=4, nhead=4, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.emb_ch = emb_ch
        self.proj = nn.Conv2d(emb_ch, emb_ch, kernel_size=patch_size, stride=patch_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_ch, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.Hp = None
        self.Wp = None

    def forward(self, x):
        # x: B,emb,H,W
        B, E, H, W = x.shape                 # save original H,W
        z = self.proj(x)  # B,emb,Hp,Wp
        B, E, Hp, Wp = z.shape
        self.Hp, self.Wp = Hp, Wp
        tokens = z.flatten(2).transpose(1,2)  # B, Npatch, E
        out = self.transformer(tokens)  # B, Npatch, E
        # reshape back
        out = out.transpose(1,2).view(B, E, Hp, Wp)
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

class UNetDecoderSmall(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,out_ch,3,padding=1)
        )
    def forward(self,x):
        d=self.down(x)
        return self.up(d)

class FullNet(nn.Module):
    def __init__(self,in_vars,out_vars,emb_ch=16,patch_size=4):
        super().__init__()
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.num_in = len(in_vars)
        self.num_out = len(out_vars)
        self.encoders = nn.ModuleDict({v:VarEncoder(1,emb_ch) for v in in_vars})
        self.processor = SimplePatchViT(emb_ch, patch_size)
        self.decoder = UNetDecoderSmall(emb_ch, self.num_out)

    def forward(self,x):
        feats=[]
        for i,v in enumerate(self.in_vars):
            xi = x[:,i:i+1,:,:]
            feats.append(self.encoders[v](xi))
        fused = torch.stack(feats,dim=0).mean(dim=0)
        processed = self.processor(fused)
        out = self.decoder(processed)
        return out


def train(model,dataloader,optimizer,criterion,device):
    model.train()
    total_loss=0.0
    for i,(xb,yb) in enumerate(dataloader):
        xb,yb=xb.to(device),yb.to(device)
        if i==0:
            print("Batch input stats:", xb.min().item(), xb.max().item(), xb.mean().item())
            print("Batch target stats:", yb.min().item(), yb.max().item(), yb.mean().item())
        optimizer.zero_grad()
        pred = model(xb)
        if torch.isnan(pred).any(): continue
        loss = criterion(pred,yb)
        if torch.isnan(loss): continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

            loss = criterion(pred, yb)
            total_loss += loss.item()

            # save for metrics
            y_true.append(yb.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Metrics (without sklearn)
    mse = np.mean((y_true.flatten() - y_pred.flatten())**2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true.flatten() - y_pred.flatten())**2)
    ss_tot = np.sum((y_true.flatten() - np.mean(y_true.flatten()))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

    return total_loss / len(dataloader), rmse, r2, y_true, y_pred



def plot_maps(y_true, y_pred, idx=0, varname="Var", outdir="outputs"):

    os.makedirs(outdir, exist_ok=True)

    # handle small batch size (avoid IndexError)
    idx = min(idx, y_true.shape[0]-1)

    true_map = y_true[idx, 0]
    pred_map = y_pred[idx, 0]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(true_map, cmap="coolwarm")
    axs[0].set_title(f"{varname} - Truth")
    plt.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(pred_map, cmap="coolwarm")
    axs[1].set_title(f"{varname} - Prediction")
    plt.colorbar(im1, ax=axs[1], fraction=0.046)

    plt.suptitle(f"Sample {idx} - {varname}")
    plt.tight_layout()

    # Save instead of show
    outfile = os.path.join(outdir, f"{varname}_sample{idx}.png")
    plt.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved plot: {outfile}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--input_vars', nargs='+', required=True)
    parser.add_argument('--target_vars', nargs='+', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lead_time', type=int, default=1)
    parser.add_argument('--level_reduce', choices=['first','mean'], default='first')
    parser.add_argument('--level_index', type=int, default=0)
    parser.add_argument('--emb_ch', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("[INFO] device:", device)

    # Dataset
    ds = WeatherDataset(args.data_dir,
                        target_vars=args.target_vars,
                        input_vars=args.input_vars,
                        level_reduce=args.level_reduce,
                        level_index=args.level_index)

    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FullNet(args.input_vars, args.target_vars, emb_ch=args.emb_ch, patch_size=args.patch_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, dl_train, optimizer, loss_fn, device)
        val_loss, val_rmse, val_r2, y_true, y_pred = evaluate(model, dl_val, loss_fn, device)
        print(f"[EPOCH {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"RMSE: {val_rmse:.4f} | R(sqr): {val_r2:.4f}")
        ckpt = os.path.join(args.output_dir, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)

    plot_maps(y_true, y_pred, idx=0, varname=args.target_vars[0])
    plot_maps(y_true, y_pred, idx=1, varname=args.target_vars[1])

if __name__=="__main__":
    main()