
"""
ResNet‑18
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
from einops import rearrange


def min_max(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min‑max normalise 1‑D signal to 0‑1."""
    return (x - x.min()) / (x.max() - x.min() + eps)


def detrend(sig: np.ndarray, order: int = 2) -> np.ndarray:
    t = np.arange(len(sig))
    return sig - np.polyval(np.polyfit(t, sig, order), t)


def bandpass(sig: np.ndarray, fs: float, f_lo: float = 0.75, f_hi: float = 2.5,
             order: int = 2) -> np.ndarray:
    b, a = butter(order, [f_lo / (0.5 * fs), f_hi / (0.5 * fs)], btype="band")
    return filtfilt(b, a, sig)

#Face crops

_mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                          max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

_ROI_LANDMARKS = {10, 338, 297, 332, 284, 251, 389, 356, 234, 93, 132, 58, 172, 162, 127}


def _crop_face(frame: np.ndarray) -> np.ndarray:
    h, w, _ = frame.shape
    res = _mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return frame  # fallback: whole image
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(int(l.x * w), int(l.y * h))
                    for i, l in enumerate(lm) if i in _ROI_LANDMARKS])
    x1, y1 = pts.min(axis=0);
    x2, y2 = pts.max(axis=0)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    size = int(1.3 * max(x2 - x1, y2 - y1))
    x1, y1 = max(cx - size // 2, 0), max(cy - size // 2, 0)
    x2, y2 = min(cx + size // 2, w - 1), min(cy + size // 2, h - 1)
    return frame[y1:y2, x1:x2]


#Model 

class ResNetRPPG(nn.Module):
    """ResNet‑18 backbone + lightweight 1‑D TCN head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        rn = models.resnet18(pretrained=pretrained)
        self.enc = nn.Sequential(*list(rn.children())[:-2])  # conv‑to‑layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Temporal conv head: 512→256→128→1
        self.tcn = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128,   1, 3, padding=1)  # no BN/ReLU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        f = self.pool(self.enc(x)).flatten(1)           # (B*T,512)
        f = rearrange(f, '(b t) c -> b c t', b=B, t=T)  # [B,512,T]
        return self.tcn(f).squeeze(1)                   # [B,T]


#Extraction for a single video#

_tf_infer = T.Compose([
    T.ToTensor(),                
    T.Resize((128, 128)),
])

def extract_rppg_signal_resnet(video_path: str,
                               model_path: str,
                               frames: int | None = None,
                               device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
                               verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Infer rPPG waveform for **an entire video**.

    Returns
    -------
    times : np.ndarray  shape=(N,)
        Timestamps in seconds (0…duration).
    signal : np.ndarray shape=(N,)
        Min‑max‑scaled predicted waveform ∈[0,1].
    """
    #Load model
    model = ResNetRPPG(pretrained=False).to(device).eval()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)

    #Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames is None or frames > tot:
        frames = tot
    tensor_buf: List[torch.Tensor] = []
    for _ in range(frames):
        ret, frm = cap.read()
        if not ret:
            break
        crop = _crop_face(frm)
        tensor_buf.append(_tf_infer(crop))
    cap.release()
    if not tensor_buf:
        raise RuntimeError("No frames extracted ― check video/codec!")

    vid = torch.stack(tensor_buf).unsqueeze(0).to(device)
    with torch.no_grad():
        wave = model(vid).squeeze(0).cpu().numpy()
    wave = min_max(wave)
    times = np.arange(len(wave)) / fps
    if verbose:
        print(f"[✓] {os.path.basename(video_path)} → {len(wave)} samples, fps={fps:.2f}")
    return times, wave

#Batch evaluation (RMSE/MAE)

def _match_pairs(root: str) -> List[Tuple[str, str]]:
    pairs = []
    for vid in glob.glob(os.path.join(root, '**', '*.avi'), recursive=True):
        # Prefer *_gt.txt, else .txt next to .avi
        gt = vid.replace('.avi', '_gt.txt')
        if not os.path.exists(gt):
            gt = os.path.splitext(vid)[0] + '.txt'
        if os.path.exists(gt):
            pairs.append((vid, gt))
    return sorted(pairs)


def process_all_videos_resnet(dataset_path: str,
                              model_path: str,
                              frames: int | None = None,
                              device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    """Mirror‑функция к вашему baseline: печатает RMSE+MAE per file + mean."""
    total_rmse, total_mae, processed = [], [], 0

    pairs = _match_pairs(dataset_path)
    if not pairs:
        raise RuntimeError(f"No <video,gt> pairs found in {dataset_path}")

    for vid_path, gt_path in pairs:
        try:
            name = os.path.basename(vid_path)
            t0 = _time.time()
            times, pred = extract_rppg_signal_resnet(vid_path, model_path, frames, device)
            with open(gt_path) as f:
                lines = f.read().splitlines()
            if len(lines) < 3:
                warnings.warn(f"[!] {gt_path} malformed (~<3 lines). Skip.")
                continue
            gt_sig = np.fromstring(lines[0], sep=' ')
            gt_tim = np.fromstring(lines[2], sep=' ')
            f_gt = interp1d(gt_tim, gt_sig, kind='linear', fill_value="extrapolate")
            gt_aligned = min_max(f_gt(times))
            #Metrics
            rmse = math.sqrt(mean_squared_error(gt_aligned, pred))
            mae = mean_absolute_error(gt_aligned, pred)
            dt = _time.time() - t0
            print(f"{name:<40} RMSE={rmse:.4f}  MAE={mae:.4f}  ({dt*1e3:.0f} ms)")
            total_rmse.append(rmse); total_mae.append(mae); processed += 1
        except Exception as e:
            warnings.warn(f"[×] {os.path.basename(vid_path)} – {e}")

    if processed:
        print("-"*62)
        print(f"Mean RMSE: {np.mean(total_rmse):.4f}\nMean MAE : {np.mean(total_mae):.4f}")
    else:
        print("[!] No video‑GT pairs processed.")

#CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rPPG extraction/eval via ResNet‑18")
    parser.add_argument('--dataset', default='test', help='path to folder with videos & *.txt ground‑truth files')
    parser.add_argument('--model',   default='ckpt_ep25.pth', help='trained .pth checkpoint')
    parser.add_argument('--frames',  type=int, default=None, help='override number of frames to process per video')
    args = parser.parse_args()
    process_all_videos_resnet(args.dataset, args.model, args.frames)
55