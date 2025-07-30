#!/usr/bin/env python3
"""
GLBファイルを読み込み、オブジェクトをY軸周りに90度回転させて保存するスクリプト

必要なライブラリ:
- pygltflib: pip install pygltflib
- numpy: pip install numpy

使用法:
python rotate_glb.py input.glb [output.glb]          # デフォルトは下90度回転
python rotate_glb.py -l input.glb [output.glb]       # 左90度回転
python rotate_glb.py -r input.glb [output.glb]       # 右90度回転
python rotate_glb.py -u input.glb [output.glb]       # 上90度回転
python rotate_glb.py -d input.glb [output.glb]       # 下90度回転
python rotate_glb.py -h input.glb [output.glb]       # 水平180度回転
python rotate_glb.py -v input.glb [output.glb]       # 天地180度回転
# output.glbを省略すると入力ファイルを上書き
"""

import sys
import argparse
import numpy as np
import subprocess
from pygltflib import GLTF2


def rotate_model(gltf, rotation_type='down'):
    """GLTFモデルを回転"""
    if rotation_type == 'left':
        # 左回転90度（Y軸周り-90度）
        rotation_matrix = np.array([
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif rotation_type == 'right':
        # 右回転90度（Y軸周り+90度）
        rotation_matrix = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif rotation_type == 'up':
        # 上回転90度（Z軸周り-90度）
        rotation_matrix = np.array([
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif rotation_type == 'down':
        # 下回転90度（X軸周り-90度）手前に倒す
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif rotation_type == 'horizontal':
        # 水平180度回転（Y軸周り180度）
        rotation_matrix = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    elif rotation_type == 'vertical':
        # 天地180度回転（Z軸周り180度）
        rotation_matrix = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported rotation type: {rotation_type}")
    
    # 各ノードの変換行列を更新
    for node in gltf.nodes:
        # 既存の行列を無視して、新しい回転行列のみを設定
        node.matrix = rotation_matrix.flatten().tolist()
        # translation, rotation, scaleをクリア
        node.translation = None
        node.rotation = None
        node.scale = None
    
    return gltf


def main():
    parser = argparse.ArgumentParser(description='GLBファイルを回転', add_help=False)
    parser.add_argument('input_file', nargs='?', help='入力GLBファイル')
    parser.add_argument('output_file', nargs='?', help='出力GLBファイル（省略時は入力ファイルを上書き）')
    parser.add_argument('-l', '--left', action='store_true', help='左90度回転')
    parser.add_argument('-r', '--right', action='store_true', help='右90度回転')
    parser.add_argument('-u', '--up', action='store_true', help='上90度回転')
    parser.add_argument('-d', '--down', action='store_true', help='下90度回転')
    parser.add_argument('-h', '--horizontal', action='store_true', help='水平180度回転')
    parser.add_argument('-v', '--vertical', action='store_true', help='天地180度回転')
    parser.add_argument('--help', action='help', help='ヘルプを表示')
    
    args = parser.parse_args()
    
    # 引数なしの場合はヘルプを表示
    if not args.input_file:
        parser.print_help()
        return
    
    # 回転タイプを決定
    rotation_options = [
        (args.left, 'left', '左90度'),
        (args.right, 'right', '右90度'),
        (args.up, 'up', '上90度'),
        (args.down, 'down', '下90度'),
        (args.horizontal, 'horizontal', '水平180度'),
        (args.vertical, 'vertical', '天地180度')
    ]
    
    selected_rotations = [opt for opt in rotation_options if opt[0]]
    
    if len(selected_rotations) == 0:
        rotation_type = 'down'
        rotation_desc = '下90度（デフォルト）'
    elif len(selected_rotations) == 1:
        rotation_type = selected_rotations[0][1]
        rotation_desc = selected_rotations[0][2]
    else:
        print("エラー: 複数の回転オプションが指定されています")
        sys.exit(1)
    
    # 出力ファイルが指定されていない場合は入力ファイルと同じにする
    output_file = args.output_file if args.output_file else args.input_file
    
    try:
        # GLBファイルを読み込み
        print(f"読み込み中: {args.input_file}")
        gltf = GLTF2.load(args.input_file)
        
        # 回転
        print(f"{rotation_desc}回転中...")
        rotated_gltf = rotate_model(gltf, rotation_type)
        
        # 回転したモデルを保存
        print(f"保存中: {output_file}")
        rotated_gltf.save(output_file)
        
        print("完了!")
        
        # ファイルを開く
        try:
            subprocess.run(["open", output_file])
        except Exception as open_error:
            print(f"ファイルオープン警告: {open_error}")
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()