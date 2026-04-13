#!/usr/bin/env python3
"""
车牌字符识别 - 全量训练版（不采样，不减少训练集）
内存需求: 约6~8GB
训练时间: 约2~4小时
"""
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.ndimage import rotate, shift
from collections import Counter
from tqdm import tqdm

DATA_DIR  = r"D:\Desktop\char_dataset"
CKPT_PATH = r"D:\Desktop\best_plate.h5"
OUTPUT_H  = r"D:\Desktop\plate_weights.h"

CHAR_H     = 40
CHAR_W     = 20
INPUT_SIZE = 800

PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
             "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
ALPHABETS = ['A','B','C','D','E','F','G','H','J','K',
             'L','M','N','P','Q','R','S','T','U','V',
             'W','X','Y','Z']
ADS       = ['A','B','C','D','E','F','G','H','J','K',
             'L','M','N','P','Q','R','S','T','U','V',
             'W','X','Y','Z',
             '0','1','2','3','4','5','6','7','8','9']
ALL_CHARS   = PROVINCES + ALPHABETS + ADS
NUM_CLASSES = len(ALL_CHARS)
print(f"总类别数: {NUM_CLASSES}")

# ============================================================
# 1. 全量加载（不限制每类数量）
# ============================================================
def load_all(data_dir):
    X, y = [], []
    print("全量加载数据（不采样）...")
    for idx in tqdm(range(NUM_CLASSES), desc="读取类别"):
        class_dir = os.path.join(data_dir, str(idx))
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith('.jpg')]
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (CHAR_W, CHAR_H))
            _, img = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            X.append(img.flatten().astype(np.float32) / 255.0)
            y.append(idx)

    X_out = np.array(X, dtype=np.float32)
    y_out = np.array(y, dtype=np.int32)
    mem_gb = X_out.nbytes / 1024**3
    print(f"加载完成: {len(X_out)} 张，内存占用: {mem_gb:.2f} GB")
    return X_out, y_out

# ============================================================
# 2. 只对少数类增强（多数类不动）
# ============================================================
def augment_single(img_flat):
    img = img_flat.reshape(CHAR_H, CHAR_W)
    if np.random.rand() < 0.6:
        angle = np.random.uniform(-12, 12)
        img = rotate(img, angle, reshape=False, cval=0)
    if np.random.rand() < 0.6:
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        img = shift(img, [dy, dx], cval=0)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.8, 1.2)
        new_h = max(4, int(CHAR_H * scale))
        new_w = max(4, int(CHAR_W * scale))
        scaled = cv2.resize(img.astype(np.float32),
                            (new_w, new_h))
        canvas = np.zeros((CHAR_H, CHAR_W), dtype=np.float32)
        y0 = max(0, (CHAR_H - new_h) // 2)
        x0 = max(0, (CHAR_W - new_w) // 2)
        sh = min(new_h, CHAR_H - y0)
        sw = min(new_w, CHAR_W - x0)
        canvas[y0:y0+sh, x0:x0+sw] = scaled[:sh, :sw]
        img = canvas
    if np.random.rand() < 0.3:
        img = np.clip(
            img + np.random.randn(CHAR_H, CHAR_W) * 0.05,
            0, 1).astype(np.float32)
    return (img > 0.5).astype(np.float32).flatten()

def augment_minority(X, y, min_samples=500):
    counts = Counter(y.tolist())
    minority = {cls: cnt for cls, cnt in counts.items()
                if cnt < min_samples}

    if not minority:
        print("所有类别样本数已足够，无需增强")
        return X, y

    print(f"对 {len(minority)} 个少数类增强到{min_samples}张...")
    X_aug, y_aug = [], []
    for cls, cnt in tqdm(minority.items(), desc="增强"):
        cls_idx = np.where(y == cls)[0]
        need = min_samples - cnt
        for _ in range(need):
            i = np.random.choice(cls_idx)
            X_aug.append(augment_single(X[i]))
            y_aug.append(cls)

    if X_aug:
        X_out = np.vstack([X, np.array(X_aug, dtype=np.float32)])
        y_out = np.concatenate([y, np.array(y_aug, dtype=np.int32)])
        idx = np.random.permutation(len(X_out))
        X_out = X_out[idx]
        y_out = y_out[idx]
        mem_gb = X_out.nbytes / 1024**3
        print(f"增强后: {len(X_out)} 张，内存: {mem_gb:.2f} GB")
        return X_out, y_out
    return X, y

# ============================================================
# 3. 模型
# ============================================================
def build_model():
    inp = keras.Input(shape=(INPUT_SIZE,))
    x = layers.Dense(256, name='d1')(inp)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, name='d2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES,
                       activation='softmax', name='d3')(x)
    return keras.Model(inp, out, name='plate_char')

# ============================================================
# 4. 主流程
# ============================================================
print("\n[1/4] 全量加载数据...")
X, y = load_all(DATA_DIR)

counts = Counter(y.tolist())
print(f"最多类: {max(counts.values())} 张")
print(f"最少类: {min(counts.values())} 张")
print(f"有样本类别数: {len(counts)}/{NUM_CLASSES}")

print("\n[2/4] 少数类增强...")
X, y = augment_minority(X, y, min_samples=500)

y_cat = keras.utils.to_categorical(y, NUM_CLASSES)
split = int(len(X) * 0.9)
X_tr,  X_val  = X[:split],     X[split:]
y_tr,  y_val  = y_cat[:split], y_cat[split:]
print(f"训练集: {len(X_tr)}, 验证集: {len(X_val)}")

# 释放原始数组节省内存
del X, y
import gc
gc.collect()

print("\n[3/4] 训练模型...")
model = build_model()
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(
        CKPT_PATH,
        monitor='val_accuracy',
        save_best_only=True, verbose=1),
]

model.fit(
    X_tr, y_tr,
    epochs=60,
    batch_size=512,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n验证集准确率: {acc*100:.2f}%")

# ============================================================
# 5. 融合BN，导出C头文件
# ============================================================
print("\n[4/4] 导出权重...")

def fuse_bn(w, b, bn):
    gamma, beta, mean, var = bn.get_weights()
    scale = gamma / np.sqrt(var + bn.epsilon)
    return (w * scale[np.newaxis,:]).astype(np.float32), \
           ((b - mean)*scale + beta).astype(np.float32)

w1,b1 = fuse_bn(*model.get_layer('d1').get_weights(),
                 model.get_layer('bn1'))
w2,b2 = fuse_bn(*model.get_layer('d2').get_weights(),
                 model.get_layer('bn2'))
w3 = model.get_layer('d3').get_weights()[0].astype(np.float32)
b3 = model.get_layer('d3').get_weights()[1].astype(np.float32)

total = sum(a.size for a in [w1,b1,w2,b2,w3,b3])
print(f"总参数: {total} ({total*4/1024:.1f} KB)")

def to_c(name, arr, cols=8):
    flat = arr.flatten().tolist()
    lines = [f"static const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), cols):
        chunk = flat[i:i+cols]
        lines.append("    " +
            ", ".join(f"{v:.6f}f" for v in chunk) + ",")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    return "\n".join(lines)

char_map_c  = f"#define PLATE_NUM_CLASSES {NUM_CLASSES}\n\n"
char_map_c += "static const char *PLATE_CHARS[] = {\n"
for c in ALL_CHARS:
    if ord(c) > 127:
        encoded = c.encode('utf-8')
        hex_str = ''.join(f'\\x{b:02x}' for b in encoded)
        char_map_c += f'    "{hex_str}",\n'
    else:
        char_map_c += f'    "{c}",\n'
char_map_c += "};\n"

header = f"""\
/*************************************************************
 * plate_weights.h
 * 网络: {INPUT_SIZE}->{w1.shape[1]}->{w2.shape[1]}
 *      ->{NUM_CLASSES}(Softmax)
 * 验证准确率: {acc*100:.2f}%
 * Flash: {total*4/1024:.1f} KB
 *************************************************************/
#ifndef __PLATE_WEIGHTS_H
#define __PLATE_WEIGHTS_H

#define PLATE_INPUT_SIZE   {INPUT_SIZE}
#define PLATE_HIDDEN1_SIZE {w1.shape[1]}
#define PLATE_HIDDEN2_SIZE {w2.shape[1]}
#define PLATE_NUM_CLASSES  {NUM_CLASSES}

{char_map_c}
{to_c("plate_w1", w1)}
{to_c("plate_b1", b1)}
{to_c("plate_w2", w2)}
{to_c("plate_b2", b2)}
{to_c("plate_w3", w3)}
{to_c("plate_b3", b3)}

#endif
"""

with open(OUTPUT_H, "w", encoding="utf-8") as f:
    f.write(header)
print(f"✅ plate_weights.h 已生成: {OUTPUT_H}")