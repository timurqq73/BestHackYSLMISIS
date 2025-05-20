"""
Архитектура: torchvision `r3d_18` (ResNet‑18 3D) без FC‑головы + upsampling +
1‑D Conv ⇒ предсказание покадрового rPPG‑сигнала.
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
import torchvision.models.video as models3d
from einops import rearrange


def min_max(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min() + eps)


def detrend(sig: np.ndarray, order: int = 2) -> np.ndarray:
    t = np.arange(len(sig))
    return sig - np.polyval(np.polyfit(t, sig, order), t)


def bandpass(sig: np.ndarray, fs: float, lo: float = 0.75, hi: float = 2.5,
             order: int = 2) -> np.ndarray:
    b, a = butter(order, [lo/(0.5*fs), hi/(0.5*fs)], btype='band')
    return filtfilt(b, a, sig)


_mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                          max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)
_ROI = {10,338,297,332,284,251,389,356,234,93,132,58,172,162,127}

def _crop_face(frame: np.ndarray) -> np.ndarray:
    h,w,_ = frame.shape
    res = _mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return frame
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(int(l.x*w), int(l.y*h)) for i,l in enumerate(lm) if i in _ROI])
    x1,y1 = pts.min(0); x2,y2 = pts.max(0)
    cx,cy = (x1+x2)//2, (y1+y2)//2
    size  = int(1.3*max(x2-x1, y2-y1))
    x1,y1 = max(cx-size//2,0), max(cy-size//2,0)
    x2,y2 = min(cx+size//2,w-1), min(cy+size//2,h-1)
    return frame[y1:y2, x1:x2]

class R3D18_RPPG(nn.Module):
    """3D‑CNN (r3d_18) → per‑frame rPPG waveform."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = models3d.r3d_18(weights=models3d.R3D_18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # без avgpool+fc
        self.head_conv = nn.Conv1d(512, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,T,H,W]
        f = self.backbone(x)              # [B,512,T',H',W']  T' = T/8
        f = f.mean([-2,-1])               # spatial GAP → [B,512,T']
        t_out = x.shape[2]
        f_up = F.interpolate(f, size=t_out, mode='linear', align_corners=False)
        wave = self.head_conv(f_up).squeeze(1)  # [B,T]
        return wave


_tf = T.Compose([T.ToTensor(), T.Resize((128,128))])

class RPPGVideoDataset3D(torch.utils.data.Dataset):
    """Возвращает клипы [C,T,H,W] + GT‑сигнал той же длины."""
    def __init__(self, root: str, frames: int = 160, jitter: bool = True):
        self.meta = []
        for vid in glob.glob(os.path.join(root, '**', '*.avi'), recursive=True):
            gt = vid.replace('.avi', '_gt.txt')
            if not os.path.exists(gt):
                gt = os.path.splitext(vid)[0] + '.txt'
            if os.path.exists(gt):
                self.meta.append((vid, gt))
        self.frames = frames
        cj = T.ColorJitter(0.1,0.1,0.1,0.1)
        self.aug = cj if jitter else nn.Identity()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        vid_path, gt_path = self.meta[idx]
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = 0 if total<=self.frames else random.randint(0,total-self.frames-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.frames):
            ret, frm = cap.read()
            frm = frm if ret else frames[-1]
            crop = _crop_face(frm)
            frames.append(self.aug(_tf(crop)))
        cap.release()
        video = torch.stack(frames)          # [T,C,H,W]
        # GT align
        lines = open(gt_path).read().splitlines()
        gt_sig = np.fromstring(lines[0],sep=' ')
        gt_tim = np.fromstring(lines[2],sep=' ')
        f = interp1d(gt_tim, gt_sig, kind='linear', fill_value="extrapolate")
        ts = np.arange(self.frames)/fps + gt_tim[0]
        target = bandpass(detrend(f(ts)), fps)
        target = min_max(target)
        return video.permute(1,0,2,3), torch.tensor(target, dtype=torch.float32), torch.tensor(fps)


#Training loop


def train_3dcnn(root='dataset', epochs=20, lr=1e-4, batch=2, frames=160):
    ds = RPPGVideoDataset3D(root, frames, jitter=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = R3D18_RPPG().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(1, epochs+1):
        model.train(); loss_hist=[]
        for vid, gt, fps in dl:
            vid, gt = vid.to(dev, non_blocking=True), gt.to(dev)
            with torch.cuda.amp.autocast():
                pred = model(vid)          # [B,T]
                loss = F.mse_loss(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(opt); opt.zero_grad(); scaler.update()
            loss_hist.append(loss.item())
        print(f'E{ep:02d}: MSE={np.mean(loss_hist):.4f}')
        torch.save(model.state_dict(), f'ckpt3d_ep{ep:02d}.pth')


#Single‑video inference function

_tf_infer = T.Compose([T.ToTensor(), T.Resize((128,128))])

def extract_rppg_signal_3dcnn(video_path: str, model_path: str,
                              device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
                              verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Извлечь rPPG с помощью 3D‑CNN на всём видео."""
    model = R3D18_RPPG(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret: break
        crop = _crop_face(frm)
        frames.append(_tf_infer(crop))
    cap.release()
    video = torch.stack(frames).permute(1,0,2,3).unsqueeze(0).to(device)
    with torch.no_grad():
        wave = model(video).squeeze(0).cpu().numpy()
    wave = min_max(wave)
    times = np.arange(len(wave))/fps
    if verbose:
        print(f"[✓] {os.path.basename(video_path)}: {len(wave)} frames, fps={fps:.2f}")
    return times, wave


#evaluation(RMSE/MAE


def _pair_list(root: str):
    pairs=[]
    for vid in glob.glob(os.path.join(root,'**','*.avi'), recursive=True):
        gt = vid.replace('.avi','_gt.txt')
        if not os.path.exists(gt):
            gt = os.path.splitext(vid)[0]+'.txt'
        if os.path.exists(gt):
            pairs.append((vid,gt))
    return sorted(pairs)


def process_all_videos_3dcnn(dataset_path: str, model_path: str,
                             device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    total_rmse,total_mae,processed=[],[],0
    for vid_path, gt_path in _pair_list(dataset_path):
        try:
            t0=_time.time()
            times,pred = extract_rppg_signal_3dcnn(vid_path, model_path, device)
            lines=open(gt_path).read().splitlines()
            if len(lines)<3: raise RuntimeError('GT file malformed')
            gt_sig=np.fromstring(lines[0],sep=' ')
            gt_tim=np.fromstring(lines[2],sep=' ')
            f_gt=interp1d(gt_tim, gt_sig, kind='linear', fill_value='extrapolate')
            gt_aligned=min_max(f_gt(times))
            rmse=math.sqrt(mean_squared_error(gt_aligned,pred))
            mae=mean_absolute_error(gt_aligned,pred)
            dt=_time.time()-t0
            print(f"{os.path.basename(vid_path):<35} RMSE={rmse:.4f} MAE={mae:.4f} ({dt*1e3:.0f} ms)")
            total_rmse.append(rmse); total_mae.append(mae); processed+=1
        except Exception as e:
            warnings.warn(f"{os.path.basename(vid_path)} → {e}")
    if processed:
        print('-'*60)
        print(f"Mean RMSE: {np.mean(total_rmse):.4f}\nMean MAE : {np.mean(total_mae):.4f}")
    else:
        print('[!] No pairs processed.')


#CLI 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D‑CNN rPPG extractor')
    parser.add_argument('--dataset', default='test', help='Folder with videos + GT .txt files')
    parser.add_argument('--model', default='ckpt3d_ep20.pth', help='Path to trained checkpoint')
    parser.add_argument('--train', action='store_true', help='Run training instead of inference')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--frames', type=int, default=160)
    args = parser.parse_args()

    if args.train:
        train_3dcnn(root=args.dataset, epochs=args.epochs, frames=args.frames)
    else:
        process_all_videos_3dcnn(args.dataset, args.model)
