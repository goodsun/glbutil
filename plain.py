#!/usr/bin/env python3
"""
GLBファイルからメッシュとテクスチャのみを抽出して別のGLBファイルにコピーするスクリプト

必要なライブラリ:
- pygltflib: pip install pygltflib

使用法:
python copy_glb_mesh_texture.py input.glb output.glb
"""

import sys
import argparse
from pygltflib import GLTF2, BufferFormat


def copy_mesh_and_texture(input_path, output_path):
    """GLBファイルからメッシュとテクスチャのみをコピー"""
    
    # 入力GLBファイルを読み込み
    gltf = GLTF2.load(input_path)
    
    # 不要な要素を削除（アニメーションなど）
    gltf.animations = None
    gltf.skins = None
    
    # カメラとライトを削除
    gltf.cameras = None
    
    # 出力ファイルに保存
    gltf.save(output_path)
    
    return gltf


def main():
    parser = argparse.ArgumentParser(description='GLBファイルからメッシュとテクスチャのみをコピー')
    parser.add_argument('input_file', help='入力GLBファイル')
    parser.add_argument('output_file', help='出力GLBファイル')
    
    args = parser.parse_args()
    
    try:
        print(f"読み込み中: {args.input_file}")
        print(f"メッシュとテクスチャを抽出中...")
        
        copy_mesh_and_texture(args.input_file, args.output_file)
        
        print(f"保存完了: {args.output_file}")
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()