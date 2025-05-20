# -*- coding: utf-8 -*-
"""
Hybrid CNN + Transformer pipeline for rPPG extraction.
"""
import os, cv2, glob, argparse, random, math, warnings, time as _time
from typing import Tuple, List

import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tvm
from einops import rearrange


#Utils

def min_max(x: np.ndarray, eps: float = 1e-9):
    return (x - x.min()) / (x.max() - x.min() + eps)

def detrend(sig: np.ndarray, order: int = 2):
    t = np.arange(len(sig))
    return sig - np.polyval(np.polyfit(t, sig, order), t)

def bandpass(sig: np.ndarray, fs: float, lo: float = 0.75, hi: float = 2.5, order: int = 2):
    b, a = butter(order, [lo/(0.5*fs), hi/(0.5*fs)], btype='band')
    return filtfilt(b, a, sig)


#Face crop
_mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
_ROI = {10,338,297,332,284,251,389,356,234,93,132,58,172,162,127}

def _crop_face(frame: np.ndarray):
    h,w,_ = frame.shape
    res = _mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return frame
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(int(l.x*w), int(l.y*h)) for i,l in enumerate(lm) if i in _ROI])
    x1,y1 = pts.min(0); x2,y2 = pts.max(0)
    cx,cy = (x1+x2)//2, (y1+y2)//2
    size = int(1.3*max(x2-x1, y2-y1))
    x1,y1 = max(cx-size//2,0), max(cy-size//2,0)
    x2,y2 = min(cx+size//2,w-1), min(cy+size//2,h-1)
    return frame[y1:y2, x1:x2]


#Model
class CNNTransformerRPPG(nn.Module):
    """EfficientNet‑B0 + Transformer Encoder → rPPG waveform."""
    def __init__(self, pretrained: bool = True, embed_dim: int = 512,
                 nhead: int = 8, layers: int = 4):
        super().__init__()
        eff = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(eff.children())[:-2])  # до ConvHead
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                                               dim_feedforward=embed_dim*4,
                                               activation='gelu', batch_first=False)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor):  # x: [B,T,C,H,W]
        B,T,C,H,W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        f = self.pool(self.backbone(x)).flatten(1)          # (B*T,1280)
        f = self.proj(f)                                    # (B*T,D)
        f = rearrange(f, '(b t) d -> t b d', b=B, t=T)      # [T,B,D]
        z = self.transformer(f)                             # [T,B,D]
        out = self.head(z).squeeze(-1)                      # [T,B]
        return rearrange(out, 't b -> b t')


#Dataset class
_tf_train = T.Compose([T.ToTensor(), T.Resize((128,128)), T.ColorJitter(0.1,0.1,0.1,0.1)])
_tf_infer = T.Compose([T.ToTensor(), T.Resize((128,128))])

class RPPGDatasetTx(torch.utils.data.Dataset):
    def __init__(self, root: str, frames: int = 200, train: bool = True):
        self.meta = []
        for vid in glob.glob(os.path.join(root,'**','*.avi'), recursive=True):
            gt = vid.replace('.avi','_gt.txt')
            if not os.path.exists(gt):
                gt = os.path.splitext(vid)[0]+'.txt'
            if os.path.exists(gt):
                self.meta.append((vid,gt))
        self.frames = frames
        self.train = train

    def __len__(self): return len(self.meta)

    def __getitem__(self, idx):
        vid_path, gt_path = self.meta[idx]
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = 0 if total<=self.frames else random.randint(0, total-self.frames-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        tf = _tf_train if self.train else _tf_infer
        for _ in range(self.frames):
            ret, frm = cap.read()
            frm = frm if ret else frames[-1]
            frames.append(tf(_crop_face(frm)))
        cap.release()
        video = torch.stack(frames)             # [T,C,H,W]
        # load GT
        lines = open(gt_path).read().splitlines()
        gt_sig = np.fromstring(lines[0], sep=' ')
        gt_tim = np.fromstring(lines[2], sep=' ')
        f_gt = interp1d(gt_tim, gt_sig, kind='linear', fill_value='extrapolate')
        ts = np.arange(self.frames)/fps + gt_tim[0]
        target = min_max(bandpass(detrend(f_gt(ts)), fps))
        return video, torch.tensor(target, dtype=torch.float32), torch.tensor(fps)


#Training


def train_cnntx(root='dataset', epochs=30, lr=3e-4, batch=4, frames=200):
    ds = RPPGDatasetTx(root, frames, train=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     num_workers=4, pin_memory=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNTransformerRPPG().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(1, epochs+1):
        model.train(); loss_hist=[]
        for vid, gt, _ in dl:
            vid, gt = vid.to(dev, non_blocking=True), gt.to(dev)
            with torch.cuda.amp.autocast():
                pred = model(vid)          # [B,T]
                loss = F.mse_loss(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(opt); opt.zero_grad(); scaler.update()
            loss_hist.append(loss.item())
        print(f'E{ep:02d}: MSE={np.mean(loss_hist):.4f}')
        torch.save(model.state_dict(), f'ckpt_cnntx_ep{ep:02d}.pth')


#Inference util 

def extract_rppg_signal_cnntx(video_path: str, model_path: str,
                              device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
                              verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    model = CNNTransformerRPPG(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret: break
        frames.append(_tf_infer(_crop_face(frm)))
    cap.release()
    vid = torch.stack(frames).unsqueeze(0).to(device)  # [1,T,C,H,W]
    with torch.no_grad():
        wave = model(vid).squeeze(0).cpu().numpy()
    wave = min_max(wave)
    times = np.arange(len(wave))/fps
    if verbose:
        print(f"[✓] {os.path.basename(video_path)} → {len(wave)} frames, fps={fps:.2f}")
    return times, wave


#Batch evaluation (RMSE/MAE)

def _pairs(root: str):
    out=[]
    for vid in glob.glob(os.path.join(root,'**','*.avi'), recursive=True):
        gt=vid.replace('.avi','_gt.txt')
        if not os.path.exists(gt):
            gt=os.path.splitext(vid)[0]+'.txt'
        if os.path.exists(gt):
            out.append((vid,gt))
    return sorted(out)

def process_all_videos_cnntx(dataset_path: str, model_path: str,
                             device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    rmse_all, mae_all, processed = [], [], 0
    for vid_path, gt_path in _pairs(dataset_path):
        try:
            t0=_time.time()
            times, pred = extract_rppg_signal_cnntx(vid_path, model_path, device)
            lines = open(gt_path).read().splitlines()
            if len(lines)<3: raise RuntimeError('bad GT')
            gt_sig = np.fromstring(lines[0],sep=' ')
            gt_tim = np.fromstring(lines[2],sep=' ')
            gt_func = interp1d(gt_tim, gt_sig, kind='linear', fill_value='extrapolate')
            gt_aligned = min_max(gt_func(times))
            rmse = math.sqrt(mean_squared_error(gt_aligned, pred))
            mae  = mean_absolute_error(gt_aligned, pred)
            dt=_time.time()-t0
            print(f"{os.path.basename(vid_path):<35} RMSE={rmse:.4f} MAE={mae:.4f} ({dt*1e3:.0f} ms)")
            rmse_all.append(rmse); mae_all.append(mae); processed+=1
        except Exception as e:
            warnings.warn(f"{os.path.basename(vid_path)} → {e}")
    if processed:
        print('-'*60)
        print(f"Mean RMSE: {np.mean(rmse_all):.4f}\nMean MAE : {np.mean(mae_all):.4f}")
    else:
        print('[!] No video/GT pairs processed.')


#CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN+Transformer rPPG extractor')
    parser.add_argument('--dataset', default='test', help='Folder with videos & GT')
    parser.add_argument('--model', default='ckpt_cnntx_ep20.pth', help='Checkpoint path')
    parser.add_argument('--train', action='store_true', help='Train instead of infer')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--frames', type=int, default=200)
    args = parser.parse_args()

    if args.train:
        train_cnntx(root=args.dataset, epochs=args.epochs, frames=args.frames)
    else:
        process_all_videos_cnntx(args.dataset, args.model)
