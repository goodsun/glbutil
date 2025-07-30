# GLB Utility Scripts マニュアル

GLBファイル（3Dモデル）を操作するためのPythonスクリプト集です。

## 必要な環境

### 共通の依存ライブラリ
```bash
pip install trimesh numpy pillow pygltflib scipy
```

## スクリプト一覧

### 1. create.py - 画像から3Dオブジェクトを生成

画像ファイルから様々な形状の3Dオブジェクト（GLBファイル）を生成します。

#### 使用方法
```bash
python create.py <画像ファイル> [形状]
```

#### 形状オプション
- `lens` (デフォルト) - 凸レンズ形状
- `coin` - コイン形状（ベベル付き円柱）
- `cube` - キューブ形状
- `ball` - ボール形状
- `tray` - トレイ形状（コインを90度倒した形）
- `cap` - キャップ形状（レンズを90度倒した形）

#### 使用例
```bash
# 凸レンズ形状（デフォルト）
python create.py image.png

# コイン形状
python create.py image.png coin

# キューブ形状
python create.py image.png cube
```

#### 特徴
- 入力画像がテクスチャとして3Dオブジェクトに適用されます
- 出力ファイル名は`{入力画像名}_{形状}.glb`となります
- 生成後、自動的にファイルが開かれます（macOSの場合）

### 2. opt.py - GLBファイルの最適化

GLBファイルのサイズを削減し、パフォーマンスを向上させます。

#### 使用方法
```bash
python opt.py <input.glb> [mode] [target_ratio]
```

#### モードオプション
- `normal` (デフォルト) - 安全最適化（重複頂点削除、テクスチャ保持）
- `aggressive` - 適度軽量化（WebP85%品質）
- `ultra` - 積極軽量化（WebP70%品質+75%リサイズ）
- `vertex` - 頂点削除特化（未使用・重複・孤立頂点を徹底削除）
- `repair` - 穴埋め特化（メッシュの穴を検出・修復）

#### パラメータ
- `target_ratio`: 0.05-1.0の範囲（デフォルト: normal=0.7, aggressive=0.3, ultra=0.1）

#### 使用例
```bash
# 通常の最適化
python opt.py model.glb

# 積極的な軽量化
python opt.py model.glb aggressive

# 超軽量化（ファイルサイズ重視）
python opt.py model.glb ultra 0.1

# 頂点のクリーンアップ
python opt.py model.glb vertex

# メッシュの穴を修復
python opt.py model.glb repair
```

#### 主な機能
- **テクスチャ圧縮**: WebP形式での圧縮、解像度調整
- **ジオメトリ最適化**: 重複頂点の統合、未使用頂点の削除
- **メッシュ修復**: 穴の検出と自動修復
- **メタデータ削除**: 不要な情報を削除してファイルサイズを削減

### 3. plain.py - メッシュとテクスチャの抽出

GLBファイルからメッシュとテクスチャのみを抽出し、アニメーションやカメラなどの余分な要素を削除します。

#### 使用方法
```bash
python plain.py input.glb output.glb
```

#### 削除される要素
- アニメーション
- スキン（ボーン）
- カメラ
- ライト（ノードに含まれる場合）

#### 使用例
```bash
python plain.py animated_model.glb static_model.glb
```

### 4. rotate.py - GLBモデルの回転

GLBファイル内の3Dモデルを指定した方向に回転させます。

#### 使用方法
```bash
python rotate.py [オプション] input.glb [output.glb]
```

出力ファイルを省略すると入力ファイルを上書きします。

#### 回転オプション
- `-l, --left` - 左90度回転（Y軸周り）
- `-r, --right` - 右90度回転（Y軸周り）
- `-u, --up` - 上90度回転（Z軸周り）
- `-d, --down` - 下90度回転（X軸周り）※デフォルト
- `-h, --horizontal` - 水平180度回転（Y軸周り）
- `-v, --vertical` - 天地180度回転（Z軸周り）

#### 使用例
```bash
# 下90度回転（デフォルト）
python rotate.py model.glb

# 左90度回転して新しいファイルに保存
python rotate.py -l model.glb model_left.glb

# 水平180度回転（上書き）
python rotate.py -h model.glb

# 右90度回転
python rotate.py -r model.glb rotated.glb
```

## トラブルシューティング

### 共通の問題

1. **ライブラリがインストールされていない**
   ```bash
   pip install -r requirements.txt
   ```

2. **ファイルが開かない（macOS以外）**
   - スクリプト内の`open`コマンドはmacOS用です
   - WindowsやLinuxでは手動でファイルを開いてください

3. **メモリ不足エラー**
   - 大きなGLBファイルを処理する際に発生する可能性があります
   - `opt.py`の軽量化モードを使用してファイルサイズを削減してください

### opt.pyの注意事項

- `normal`モードは最も安全ですが、圧縮率は控えめです
- `ultra`モードは大幅にサイズを削減しますが、品質が低下する可能性があります
- `repair`モードはメッシュに問題がある場合のみ使用してください
- 位置情報やカメラ情報は保持されます

## 活用例

### ワークフロー例1: 画像から最適化された3Dモデルを作成
```bash
# 1. 画像からコイン型3Dモデルを生成
python create.py logo.png coin

# 2. 生成されたモデルを最適化
python opt.py logo_coin.glb aggressive

# 3. 必要に応じて回転
python rotate.py -r logo_coin_aggressive.glb
```

### ワークフロー例2: 既存モデルのクリーンアップ
```bash
# 1. アニメーションを削除
python plain.py animated_model.glb static_model.glb

# 2. 頂点をクリーンアップ
python opt.py static_model.glb vertex

# 3. さらに軽量化
python opt.py static_model_vertex.glb ultra
```