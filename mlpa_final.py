import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from PIL import Image
import matplotlib.pyplot as plt

# CSVファイルのパス
csv_file_path = './combined_output.csv'
image_folder_path = '.'  # 画像フォルダーのパスを指定

# CSVファイルの読み込み
data = pd.read_csv(csv_file_path)

# ラベルデータのone-hotエンコーディング
label_encoders = {}
for col in ['sire', 'bms', 'sex', 'hair color']:  # 各ラベル列に対して個別にエンコーダーを作成
    label_encoder = OneHotEncoder(sparse_output=False)
    labels_encoded = label_encoder.fit_transform(data[col].values.reshape(-1, 1))
    data = pd.concat([data, pd.DataFrame(labels_encoded, columns=label_encoder.get_feature_names_out())], axis=1)
    label_encoders[col] = label_encoder

# ターゲットデータの正規化
target_scaler = StandardScaler()
data.iloc[:, 4] = target_scaler.fit_transform(data.iloc[:, [4]]).astype(data.iloc[:, 4].dtype)

class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        img_path = img_path.lstrip('/')
        img_path = img_path.replace('/', '\\')
        image = Image.open(img_path).convert('RGB')  # 画像の読み込み
        if self.transform:
            image = self.transform(image)
        
        # 各ラベルのone-hotベクトルを取得
        label_tensors = []
        for col, encoder in label_encoders.items():
            label = self.data.iloc[idx][col]
            label_encoded = encoder.transform([[label]])[0]
            label_tensor = torch.tensor(label_encoded, dtype=torch.float32)
            label_tensors.append(label_tensor)
        
        labels = torch.cat(label_tensors, dim=0)  # すべてのラベルを一つのテンソルに結合
        
        target = self.data.iloc[idx, 4]  # ターゲットデータを取得
        target = torch.tensor(target, dtype=torch.float32)
        
        return image, labels, target

# 画像サイズの変換や正規化のための変換
transform = transforms.Compose([
    transforms.Resize((305, 395)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データセットとデータローダーの作成
dataset = CustomDataset(dataframe=data, img_dir=image_folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class CombinedModel(nn.Module):
    def __init__(self, num_labels):
        super(CombinedModel, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # ResNetの最終層を無効化
        
        self.mlp = nn.Sequential(
            nn.Linear(num_labels, 128),
            nn.BatchNorm1d(128),  # バッチ正規化
            nn.ReLU(),
            nn.Dropout(p=0.4),  # ドロップアウト
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),  # バッチ正規化
            nn.ReLU(),
            nn.Dropout(p=0.4),  # ドロップアウト
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),  # バッチ正規化
            nn.ReLU(),
            nn.Dropout(p=0.4)  # ドロップアウト
        )
        
        self.fc = nn.Linear(2048 + 64, 1)  # ResNetの出力次元2048とMLPの出力次元64を結合
    
    def forward(self, image, labels):
        image_features = self.resnet(image)
        label_features = self.mlp(labels)
        combined_features = torch.cat((image_features, label_features), dim=1)
        output = self.fc(combined_features)
        return output

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの初期化
num_labels = sum(encoder.categories_[0].shape[0] for encoder in label_encoders.values())
model = CombinedModel(num_labels=num_labels).to(device)  # モデルをデバイスに移動
criterion = nn.MSELoss()
#optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.0025)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels, targets in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, labels)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

print("Training Finished.")

# モデルのパラメータを保存するファイルパス
model_path = './trained_model.pth'
# モデルの状態_dict()を保存
torch.save(model.state_dict(), model_path)
print(f"Model parameters saved to {model_path}")

# ---ここから実際に訓練したモデルを使って予測した結果を示す---

# 別のCSVファイルのパス
new_csv_file_path = './testData.csv'
# 新しいCSVファイルの読み込み
new_data = pd.read_csv(new_csv_file_path)
# 新しいデータに対する予測用のDatasetとDataLoaderを作成
new_dataset = CustomDataset(dataframe=new_data, img_dir=image_folder_path, transform=transform)
new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=False)
# モデルの初期化とパラメータの読み込み
model = CombinedModel(num_labels=num_labels).to(device)  # モデルの初期化
model.load_state_dict(torch.load(model_path))  # 保存したパラメータを読み込む
model.eval()  # モデルを評価モードに設定

# 予測の実行
predictions = []
with torch.no_grad():
    for images, labels, targets in new_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images, labels)
        outputs = outputs.cpu().numpy()  # GPU上で計算された結果をCPUに移動
        outputs = target_scaler.inverse_transform(outputs)  # 逆正規化

        predictions.extend(outputs.flatten())

# 予測結果を新しいデータフレームに追加
new_data['predictions'] = predictions

# 新しいデータフレームをCSVファイルに保存
predicted_csv_file_path = './predicted_data.csv'
new_data.to_csv(predicted_csv_file_path, index=False)

print(f"Predictions saved to {predicted_csv_file_path}")

# CSVファイルのパスと読み込み
predicted_csv_file_path = './predicted_data.csv'
plt_data = pd.read_csv(predicted_csv_file_path)
# 表示する画像数
num_images = len(plt_data)
# 画像の表示設定
fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(20, 5))
# 画像の読み込みと表示
for i in range(num_images):
    img_path = plt_data.iloc[i, 1]  # 2列目の画像パス
    img_path = img_path.lstrip('/')
    img_path = img_path.replace('/', '\\')
    img = Image.open(img_path)
    # リサイズしたい高さ
    new_height = 1200
    # 元のアスペクト比を保ったまま高さをリサイズ
    width_percent = (new_height / float(img.size[1]))
    new_width = int((float(img.size[0]) * float(width_percent)))
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    axes[i].imshow(resized_img)
    axes[i].axis('off')  # 軸を非表示にする

    text5 = plt_data.iloc[i, 4]  # 5列目の要素
    text8 = plt_data.iloc[i, 7]  # 8列目の要素
    error = abs(text5 - text8)
    axes[i].text(0, 1400, f'Price: {text5}', fontsize=12, ha='left')
    axes[i].text(0, 1550, f'Predict: {text8}', fontsize=12, ha='left')
    axes[i].text(0, 1700, f'Error: {error}', fontsize=12, ha='left')

plt.tight_layout()
plt.show()