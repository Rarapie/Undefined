import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import pickle
import sys
import os


# ==========================================
# 1. 模型结构定义 (必须与训练代码一致)
# ==========================================

class TissueClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Linear(64, num_classes))

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)


class TimeEncoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        layers = []
        prev_dim = input_dim
        # 兼容字典或对象访问
        layer_sizes = config['time_encoder_layers'] if isinstance(config, dict) else config.time_encoder_layers
        dropout = config['dropout_rate'] if isinstance(config, dict) else config.dropout_rate

        for i, dim in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if i < len(layer_sizes) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.encoder = nn.Sequential(*layers)
        self.output_dim = layer_sizes[-1]

    def forward(self, x):
        return self.encoder(x)


class TimeAttention(nn.Module):
    def __init__(self, time_dim, context_dim):
        super().__init__()
        self.input_dim = time_dim + context_dim
        self.attention = nn.Sequential(nn.Linear(self.input_dim, time_dim), nn.Sigmoid())

    def forward(self, time_feat, context):
        combined = torch.cat([time_feat, context], dim=1)
        attention_weights = self.attention(combined)
        return time_feat * attention_weights


class MultiTaskSAE(nn.Module):
    def __init__(self, tissue_model, time_encoder, num_tissue_classes, config):
        super().__init__()
        self.tissue_model = tissue_model
        self.time_encoder = time_encoder
        tissue_output_dim = num_tissue_classes

        self.time_attention = TimeAttention(
            time_dim=time_encoder.output_dim,
            context_dim=tissue_output_dim
        )

        reg_input_dim = time_encoder.output_dim + tissue_output_dim
        layers = []
        prev_dim = reg_input_dim

        reg_layers = config['regressor_layers'] if isinstance(config, dict) else config.regressor_layers
        dropout = config['dropout_rate'] if isinstance(config, dict) else config.dropout_rate

        for i, dim in enumerate(reg_layers[:-1]):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, reg_layers[-1]))
        self.regressor = nn.Sequential(*layers)

    def forward(self, tissue_feat, time_feat):
        tissue_cls = self.tissue_model(tissue_feat)
        tissue_prob = torch.softmax(tissue_cls, dim=1)
        time_latent = self.time_encoder(time_feat)
        enhanced_time = self.time_attention(time_latent, tissue_prob)
        reg_input = torch.cat([enhanced_time, tissue_prob], dim=1)
        reg_output = self.regressor(reg_input)
        return tissue_cls, reg_output


# ==========================================
# 2. 预测逻辑
# ==========================================

def load_data_and_predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载元数据 (Scaler, 列名等)
    print(f"Loading metadata from {args.meta}...")
    with open(args.meta, 'rb') as f:
        meta = pickle.load(f)

    # 2. 读取用户输入 (无标签，第一列ID)
    print("Reading input files...")
    try:
        time_df = pd.read_csv(args.input_time, index_col=0)
        tissue_df = pd.read_csv(args.input_tissue, index_col=0)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # 3. 对齐样本 ID (取交集)
    common_ids = time_df.index.intersection(tissue_df.index)
    if len(common_ids) == 0:
        print("Error: No common IDs found between the two input files.")
        return

    print(f"Found {len(common_ids)} common samples for prediction.")
    time_df = time_df.loc[common_ids]
    tissue_df = tissue_df.loc[common_ids]

    # 4. 特征对齐与标准化
    # 强制按照训练时的列顺序提取数据，如果缺少列则报错，如果列序乱了会自动调整
    try:
        X_time_raw = time_df[meta['time_features_cols']].values
        X_tissue_raw = tissue_df[meta['tissue_features_cols']].values
    except KeyError as e:
        print(f"Error: Input files are missing features required by the model.\nMissing: {e}")
        return

    # 使用保存的Scaler进行标准化
    X_time = meta['time_scaler'].transform(X_time_raw)
    X_tissue = meta['tissue_scaler'].transform(X_tissue_raw)

    # 转换为 Tensor
    time_tensor = torch.FloatTensor(X_time).to(device)
    tissue_tensor = torch.FloatTensor(X_tissue).to(device)

    # 5. 加载模型
    # 这里需要硬编码最佳配置，或者读取外部配置。
    # 根据您的要求“基于最优模型”，我们可以在这里直接写死最优参数，或者从json读取。
    # 假设 args.config 传入了最佳结构参数
    import json
    with open(args.config, 'r') as f:
        model_config = json.load(f)

    print(f"Loading model from {args.model}...")

    tissue_input_dim = len(meta['tissue_features_cols'])
    time_input_dim = len(meta['time_features_cols'])
    num_classes = meta['num_tissue_classes']

    tissue_model = TissueClassifier(tissue_input_dim, num_classes)
    time_encoder = TimeEncoder(time_input_dim, model_config)
    model = MultiTaskSAE(tissue_model, time_encoder, num_classes, model_config)

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
    except Exception as e:
        print("Error loading model weights. Ensure config matches the pth file structure.")
        print(e)
        return

    model.to(device)
    model.eval()

    # 6. 推理
    print("Predicting...")
    batch_size = 32
    all_tissue_preds = []
    all_time_preds = []

    with torch.no_grad():
        for i in range(0, len(common_ids), batch_size):
            b_tissue = tissue_tensor[i:i + batch_size]
            b_time = time_tensor[i:i + batch_size]

            tissue_out, time_out = model(b_tissue, b_time)

            all_tissue_preds.extend(tissue_out.argmax(1).cpu().numpy())
            all_time_preds.extend(time_out.cpu().numpy().flatten())

    # 7. 结果反归一化/解码
    pred_tissue_labels = meta['tissue_label_encoder'].inverse_transform(all_tissue_preds)
    pred_time_values = meta['time_label_scaler'].inverse_transform(np.array(all_time_preds).reshape(-1, 1)).flatten()

    # 8. 保存
    results = pd.DataFrame({
        'ID': common_ids,
        'Predicted_Tissue': pred_tissue_labels,
        'Predicted_Time_Hours': pred_time_values
    })

    results.to_csv(args.output, index=False)
    print(f"✅ Success! Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA Multi-Task Prediction Tool")

    # 输入文件
    parser.add_argument("--input_time", required=True, help="Input CSV for Time features (No label, col 0 is ID)")
    parser.add_argument("--input_tissue", required=True, help="Input CSV for Tissue features (No label, col 0 is ID)")

    # 核心文件
    parser.add_argument("--model", default="best_model.pth", help="Path to trained model (.pth)")
    parser.add_argument("--meta", default="model_meta.pkl", help="Path to model metadata (.pkl)")
    parser.add_argument("--config", default="config.json", help="Path to model architecture config (.json)")

    # 输出
    parser.add_argument("--output", default="predictions.csv", help="Output filename")

    args = parser.parse_args()

    load_data_and_predict(args)