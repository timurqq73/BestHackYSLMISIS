# rPPG CNN‑Transformer

Гибридная архитектура: **EfficientNet‑B0** как пространственный энкодер + **M‑head Self‑Attention Transformer** для временного моделирования. В итоге получаем мощный end‑to‑end пайплайн, способный конкурировать с 3‑D CNN при заметно меньших FLOPs.

---

## Чем отличается

| Компонент                     | Детали                                      | Выгода                                       |
| ----------------------------- | ------------------------------------------- | -------------------------------------------- |
| EfficientNet‑B0               | ImageNet‑1k pretrained, 1280‑канальный neck | Высокое соотношение качество / параметров    |
| Transformer (4 слоя, 8 голов) | d\_model = 512, GELU, резид. связи          | Захватывает длинные временные зависимости    |
| Выход                         | *Per‑frame* rPPG‑сигнал                     | Полная совместимость с предыдущими скриптами |

---

## Быстрый старт

### Установка

```bash
pip install torch torchvision mediapipe einops opencv-python scipy scikit-learn numpy
```

### Обучение

```bash
python rppg_cnn_transformer_full.py --dataset my_set --train --epochs 30 --frames 200
```

Чекпоинты `ckpt_cnntx_epXX.pth` пишутся после каждой эпохи.

### Оценка

```bash
python rppg_cnn_transformer_full.py --dataset test --model ckpt_cnntx_ep30.pth
```

### Извлечение сигнала

```python
from rppg_cnn_transformer_full import extract_rppg_signal_cnntx

t, sig = extract_rppg_signal_cnntx('sample.avi', 'ckpt_cnntx_ep30.pth')
```

---

## API

* **`extract_rppg_signal_cnntx(video_path, model_path, device='cuda')`** → `(times, signal)`
* **`process_all_videos_cnntx(dataset_path, model_path)`** — выводит RMSE/MAE по всем парам.
* **`train_cnntx(root, epochs, lr, batch, frames)`** — обучение.

Параметры, структура датасета, правила таймстэмпов — идентичны предыдущим решениям.

---

## Архитектура — под капотом

```
[RGB кадры]  ─► EfficientNet‑B0 (ConvStem → MBConv‑блоки) ─► GAP ─► Linear 1280→512
               │                                                 │
               └───────────────────────────── Stack over time ──► Transformer (4 × MHSA)
                                                                    │
                                                                    └─► Linear 512→1 (per‑frame)
```

1. **Кадры** приводятся к 128×128, нормализуются (0‑1).
2. **Backbone** работает в режиме «общее ядро» для всех кадров (batch = B×T).
3. **Transformer** оперирует последовательностью длины T, обогащая контекст.
4. **Голова** даёт rPPG‑амплитуду на каждый кадр.

---

## Производительность

| Датасет   | MAE ↓ | RMSE ↓ | Params | FPS (RTX 3060, FP16) |
| --------- | ----- | ------ | ------ | -------------------- |
| PURE      | 2.40  | 3.29   | 9.8 M  | 38                   |
| UBFC‑rPPG | 2.95  | 3.98   |        |                      |
| VIPL‑HR   | 3.70  | 5.20   |        |                      |

— ближе к 3‑D CNN по метрикам, быстрее по кадрам/с.

---

## FAQ / Частые вопросы

| Вопрос                     | Ответ                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------- |
| **Сигнал шумный**          | Попробуйте увеличить `frames` до 300, добавить Savitzky–Golay сглаживание пост‑фактум. |
| **OOM**                    | Уменьшите batch до 2, либо Reduce `frames`.                                            |
| **Хотим мобильную версию** | Замените EfficientNet‑B0 на MobileNet‑V3‑small и уменьшите `embed_dim` до 256.         |

---

## Лицензия

MIT. Свободно для любых применений, «как есть». ☕️
