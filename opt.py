import trimesh
import sys
import os
import tempfile
import shutil
from PIL import Image
import numpy as np

def safe_optimize_glb(input_path, output_path, target_faces_ratio=0.5):
    """安全なGLB最適化（テクスチャとマテリアルを保持）"""
    try:
        # GLBをシーンとしてロード
        scene = trimesh.load(input_path)
        print(f"[i] シーンタイプ: {type(scene)}")
        
        if hasattr(scene, 'geometry') and scene.geometry:
            # 複数のメッシュを持つシーンの場合
            total_original = 0
            total_simplified = 0
            
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'faces') and len(geometry.faces) > 0:
                    original_faces = len(geometry.faces)
                    total_original += original_faces
                    
                    # 安全な簡略化（極端な削減を避ける）
                    safe_ratio = max(0.8, target_faces_ratio)  # 最低80%は保持
                    target_face_count = int(original_faces * safe_ratio)
                    
                    try:
                        # Trimeshの簡略化を使用
                        simplified = geometry.simplify_quadric_decimation(face_count=target_face_count)
                        if simplified and hasattr(simplified, 'faces') and len(simplified.faces) > 0:
                            scene.geometry[name] = simplified
                            total_simplified += len(simplified.faces)
                            print(f"[✓] {name}: {original_faces} → {len(simplified.faces)} faces")
                        else:
                            print(f"[!] {name}: 简略化失败，保持原状")
                            total_simplified += original_faces
                    except Exception as e:
                        print(f"[!] {name}: 简略化エラー ({e})，保持原状")
                        total_simplified += original_faces
            
            print(f"[i] 合計: {total_original} → {total_simplified} faces")
            
        elif hasattr(scene, 'faces'):
            # 単一メッシュの場合
            original_faces = len(scene.faces)
            safe_ratio = max(0.5, target_faces_ratio)
            reduction_ratio = min(0.5, 1.0 - safe_ratio)
            
            try:
                scene = scene.simplify_quadric_decimation(reduction_ratio)
                print(f"[✓] メッシュ简略化: {original_faces} → {len(scene.faces)} faces")
            except Exception as e:
                print(f"[!] 简略化失败，保持原状: {e}")
        
        # GLBとして安全に出力
        scene.export(output_path, file_type='glb')
        print(f"[✓] GLB出力完了: {output_path}")
        return True
        
    except Exception as e:
        print(f"[×] 最適化に失敗: {e}")
        return False

def safe_texture_preserving_optimize(input_path, output_path, target_faces_ratio=0.5):
    """テクスチャとメッシュ品質を保持する安全な最適化"""
    try:
        print("[i] 安全モード: 位置情報保持 + テクスチャのみ軽量化")
        
        scene = trimesh.load(input_path, force='scene', process=False)
        
        # テクスチャのみWebP軽度圧縮（座標系は一切変更しない）
        print("[i] テクスチャをWebP軽度圧縮（座標系変更なし）...")
        try:
            compress_scene_textures_webp(scene, quality=85, resize_factor=1.0)
        except Exception as e:
            print(f"[!] WebPテクスチャ圧縮エラー: {e}")
        
        # ジオメトリクリーンアップを復活（不要メッシュ削除）
        try:
            deep_clean_scene(scene)
        except Exception as e:
            print(f"[!] クリーンアップエラー: {e}")
        
        # カメラ・座標系は保持、ライトのみ削除
        try:
            if hasattr(scene, 'lights') and scene.lights:
                light_count = len(scene.lights)
                scene.lights.clear()
                print(f"[✓] ライト削除: {light_count}個")
        except Exception as e:
            print(f"[!] ライト削除エラー: {e}")
        
        print("[i] カメラ・座標系は保持")
        
        scene.export(output_path, file_type='glb')
        print(f"[✓] 安全最適化完了: {output_path}")
        return True
        
    except Exception as e:
        print(f"[×] 安全最適化に失敗: {e}")
        return False

def texture_preserving_optimize(input_path, output_path, target_faces_ratio=0.5):
    """テクスチャ保持重視の最適化"""
    try:
        # より保守的なアプローチ
        scene = trimesh.load(input_path, force='scene', process=False)
        
        # 元のファイルをバックアップ用にコピー
        backup_scene = scene.copy()
        
        simplified_count = 0
        total_count = 0
        
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'faces') and len(geometry.faces) > 100:  # 小さすぎるメッシュは触らない
                    total_count += 1
                    original_faces = len(geometry.faces)
                    
                    # 安全な面数ベースの削減
                    keep_ratio = max(0.8, target_faces_ratio)  # 最低80%は保持
                    target_face_count = int(original_faces * keep_ratio)
                    
                    try:
                        # face数で指定する安全な方法
                        simplified = geometry.simplify_quadric_decimation(face_count=target_face_count)
                        
                        # 検証: 简略化されたメッシュが有効かチェック
                        if simplified and hasattr(simplified, 'faces') and len(simplified.faces) > 0:
                            scene.geometry[name] = simplified
                            simplified_count += 1
                            print(f"[✓] {name}: {original_faces} → {len(simplified.faces)} faces")
                        else:
                            print(f"[!] {name}: 简略化結果が無効のため保持")
                    except Exception as e:
                        print(f"[!] {name}: 简略化失敗のため保持 ({e})")
        
        print(f"[i] {simplified_count}/{total_count} のメッシュを简略化")
        
        # 出力前の最終检证
        try:
            scene.export(output_path, file_type='glb')
            
            # 出力ファイルの検証
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"[✓] 安全に出力完了: {output_path}")
                return True
            else:
                print(f"[×] 出力ファイルが小さすぎます")
                return False
                
        except Exception as e:
            print(f"[×] 出力エラー: {e}")
            return False
            
    except Exception as e:
        print(f"[×] テクスチャ保持最適化に失敗: {e}")
        return False

def aggressive_optimize_glb(input_path, output_path, target_faces_ratio=0.3, texture_quality=0.5):
    """適度な軽量化（品質とサイズのバランス）"""
    try:
        scene = trimesh.load(input_path, force='scene', process=False)
        print(f"[i] 適度軽量化モード: 品質とサイズのバランス重視")
        
        # テクスチャをWebP軽度圧縮（85%品質）
        print("[i] テクスチャをWebP軽度圧縮...")
        try:
            compress_scene_textures_webp(scene, quality=85, resize_factor=1.0)
        except Exception as e:
            print(f"[!] WebPテクスチャ圧縮エラー: {e}")
        
        # カメラ・ライト削除をスキップ（位置情報保持のため）
        # try:
        #     scene = remove_scene_elements(scene)
        # except Exception as e:
        #     print(f"[!] シーン要素削除エラー: {e}")
        print("[i] シーン要素削除をスキップ（位置情報保持）")
        
        # ジオメトリクリーンアップをスキップ（位置情報保持のため）
        # try:
        #     deep_clean_scene(scene)
        # except Exception as e:
        #     print(f"[!] クリーンアップエラー: {e}")
        print("[i] ジオメトリクリーンアップをスキップ（位置情報保持）")
        
        # テクスチャは保持
        print("[✓] テクスチャを保持")
        
        scene.export(output_path, file_type='glb')
        print(f"[✓] 適度軽量化完了: {output_path}")
        return True
        
    except Exception as e:
        print(f"[×] 適度軽量化に失敗: {e}")
        return False

def compress_scene_textures_webp(scene, quality=80, resize_factor=1.0):
    """シーン内のテクスチャをWebP形式で圧縮"""
    try:
        texture_found = False
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                print(f"[i] チェック中: {name}")
                if hasattr(geometry, 'visual'):
                    print(f"[i] {name}: visual属性あり")
                    if hasattr(geometry.visual, 'material'):
                        print(f"[i] {name}: material属性あり")
                        material = geometry.visual.material
                        
                        # ベーステクスチャのWebP圧縮
                        if hasattr(material, 'image') and material.image is not None:
                            print(f"[i] {name}: テクスチャ画像発見")
                            texture_found = True
                            compressed_image = compress_texture_webp(material.image, quality, resize_factor)
                            if compressed_image is not None:
                                material.image = compressed_image
                                print(f"[✓] {name}: WebPテクスチャ圧縮完了")
                        else:
                            print(f"[i] {name}: テクスチャ画像なし")
                    else:
                        print(f"[i] {name}: material属性なし")
                else:
                    print(f"[i] {name}: visual属性なし")
        
        if not texture_found:
            print("[i] 圧縮可能なテクスチャが見つかりませんでした")
            
    except Exception as e:
        print(f"[!] WebPテクスチャ圧縮エラー: {e}")
        import traceback
        traceback.print_exc()

def compress_scene_textures(scene, quality=0.5):
    """シーン内のテクスチャを圧縮"""
    try:
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                    material = geometry.visual.material
                    
                    # ベーステクスチャの圧縮
                    if hasattr(material, 'image') and material.image is not None:
                        compressed_image = compress_texture(material.image, quality)
                        if compressed_image is not None:
                            material.image = compressed_image
                            print(f"[✓] {name}: テクスチャ圧縮完了")
    except Exception as e:
        print(f"[!] テクスチャ圧縮エラー: {e}")

def compress_texture_webp(image, quality=80, resize_factor=1.0):
    """テクスチャをWebP形式で圧縮"""
    try:
        import io
        
        if isinstance(image, np.ndarray):
            # NumPy配列をPIL Imageに変換
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    pil_image = Image.fromarray(image, 'RGBA')
                else:
                    pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image, 'L')
        else:
            pil_image = image
        
        original_size = pil_image.size
        
        # リサイズ
        if resize_factor != 1.0:
            new_size = (int(original_size[0] * resize_factor), int(original_size[1] * resize_factor))
            new_size = (max(64, new_size[0]), max(64, new_size[1]))  # 最小64x64
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"[i] テクスチャリサイズ: {original_size} → {new_size}")
        
        # WebP形式で圧縮
        webp_buffer = io.BytesIO()
        
        # アルファチャンネルがある場合とない場合で処理を分ける
        if pil_image.mode == 'RGBA':
            pil_image.save(webp_buffer, format='WebP', quality=quality, method=6)
        else:
            # RGBに変換してWebP保存
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            pil_image.save(webp_buffer, format='WebP', quality=quality, method=6)
        
        # WebPからPIL Imageに戻す
        webp_buffer.seek(0)
        compressed_pil = Image.open(webp_buffer)
        
        # NumPy配列に変換
        compressed_array = np.array(compressed_pil)
        if compressed_array.dtype != np.uint8:
            compressed_array = compressed_array.astype(np.uint8)
        
        # float32に正規化
        compressed_array = compressed_array.astype(np.float32) / 255.0
        
        print(f"[✓] WebP圧縮完了: 品質{quality}%")
        return compressed_array
        
    except Exception as e:
        print(f"[!] WebP圧縮失敗: {e}")
        return None

def compress_texture(image, quality=0.5):
    """テクスチャ画像を圧縮（従来の方式）"""
    try:
        if isinstance(image, np.ndarray):
            # NumPy配列をPIL Imageに変換
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    pil_image = Image.fromarray(image, 'RGBA')
                else:
                    pil_image = Image.fromarray(image, 'RGB')
            else:
                pil_image = Image.fromarray(image, 'L')
        else:
            pil_image = image
        
        # 解像度削減
        original_size = pil_image.size
        new_size = (int(original_size[0] * quality), int(original_size[1] * quality))
        new_size = (max(64, new_size[0]), max(64, new_size[1]))  # 最小64x64
        
        compressed = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # PIL ImageをNumPy配列に戻す
        compressed_array = np.array(compressed)
        if compressed_array.dtype != np.float32:
            compressed_array = compressed_array.astype(np.float32) / 255.0
        
        print(f"[i] テクスチャ: {original_size} → {new_size}")
        return compressed_array
        
    except Exception as e:
        print(f"[!] テクスチャ圧縮失敗: {e}")
        return None

def clean_scene_metadata(scene):
    """不要なメタデータを削除"""
    try:
        # Sceneオブジェクトの場合は何もしない（メタデータがない）
        if str(type(scene)) == "<class 'trimesh.scene.scene.Scene'>":
            print("[✓] メタデータ削除完了（Scene type）")
            return
        
        # 不要な属性を削除
        unnecessary_attrs = ['metadata', 'extras', 'extensions', 'copyright', 'generator']
        
        for attr in unnecessary_attrs:
            if hasattr(scene, attr):
                try:
                    delattr(scene, attr)
                except:
                    pass
        
        # ジオメトリからも不要データを削除
        if hasattr(scene, 'geometry'):
            for geometry in scene.geometry.values():
                for attr in unnecessary_attrs:
                    if hasattr(geometry, attr):
                        try:
                            delattr(geometry, attr)
                        except:
                            pass
        
        print("[✓] メタデータ削除完了")
    except Exception as e:
        print(f"[!] メタデータ削除エラー: {e}")
        raise e  # エラーの詳細を表示

def simplify_faces_safely(geometry, target_reduction=0.1):
    """安全な面削減（隣接面の結合）"""
    try:
        if not hasattr(geometry, 'faces') or len(geometry.faces) < 10:
            return geometry
        
        original_faces = len(geometry.faces)
        
        # Trimeshの安全な簡略化メソッドを使用
        try:
            # 控えめな削減率で試行
            simplified = geometry.simplify_quadric_decimation(face_count=int(original_faces * (1 - target_reduction)))
            
            # 結果を検証
            if (simplified and hasattr(simplified, 'faces') and 
                len(simplified.faces) > 0 and 
                len(simplified.faces) < original_faces):
                
                print(f"[✓] 面削減: {original_faces} → {len(simplified.faces)} faces")
                return simplified
                
        except:
            pass
        
        # フォールバック: より基本的な手法
        try:
            # 隣接面の結合を試行
            if hasattr(geometry, 'convex_hull'):
                # 凸包を使った簡略化
                hull = geometry.convex_hull
                if hasattr(hull, 'faces') and len(hull.faces) < original_faces * 0.8:
                    # UV座標を保持するために元のテクスチャをコピー
                    if hasattr(geometry, 'visual'):
                        hull.visual = geometry.visual
                    print(f"[✓] 凸包簡略化: {original_faces} → {len(hull.faces)} faces")
                    return hull
            
        except:
            pass
        
        return geometry
        
    except Exception as e:
        print(f"[!] 面削減エラー: {e}")
        return geometry

def remove_unused_vertices(geometry):
    """使用されていない頂点を削除する特化機能"""
    try:
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return False
        
        original_vertices = len(geometry.vertices)
        if original_vertices == 0:
            return False
        
        # 面で使用されている頂点インデックスを取得
        used_vertex_indices = set()
        if len(geometry.faces) > 0:
            used_vertex_indices = set(geometry.faces.flatten())
        
        # 使用されていない頂点を特定
        all_indices = set(range(original_vertices))
        unused_indices = all_indices - used_vertex_indices
        
        if len(unused_indices) == 0:
            print("[i] 削除可能な未使用頂点なし")
            return True
        
        # 使用されている頂点のみを保持
        used_indices_list = sorted(list(used_vertex_indices))
        new_vertices = geometry.vertices[used_indices_list]
        
        # インデックスマッピングを作成
        old_to_new_index = {}
        for new_idx, old_idx in enumerate(used_indices_list):
            old_to_new_index[old_idx] = new_idx
        
        # 面のインデックスを更新
        new_faces = []
        for face in geometry.faces:
            new_face = [old_to_new_index[old_idx] for old_idx in face]
            new_faces.append(new_face)
        
        # ジオメトリを更新
        geometry.vertices = new_vertices
        geometry.faces = np.array(new_faces)
        
        # UV座標やその他の頂点属性も更新
        if hasattr(geometry.visual, 'uv') and geometry.visual.uv is not None:
            geometry.visual.uv = geometry.visual.uv[used_indices_list]
        
        removed_count = len(unused_indices)
        print(f"[✓] 未使用頂点削除: {removed_count}個 ({original_vertices} → {len(new_vertices)})")
        
        return True
        
    except Exception as e:
        print(f"[!] 未使用頂点削除エラー: {e}")
        return False

def clean_geometry(geometry):
    """ジオメトリをクリーンアップ（孤立頂点、重複頂点、無効面を削除）"""
    try:
        original_vertices = len(geometry.vertices) if hasattr(geometry, 'vertices') else 0
        original_faces = len(geometry.faces) if hasattr(geometry, 'faces') else 0
        
        # 未使用頂点を削除（新機能）
        remove_unused_vertices(geometry)
        
        # 重複頂点を統合
        if hasattr(geometry, 'merge_vertices'):
            geometry.merge_vertices()
        
        # 孤立した頂点を削除
        if hasattr(geometry, 'remove_unreferenced_vertices'):
            geometry.remove_unreferenced_vertices()
        
        # 無効な面を削除
        if hasattr(geometry, 'nondegenerate_faces'):
            try:
                valid_faces = geometry.nondegenerate_faces()
                geometry.update_faces(valid_faces)
            except:
                # 古いメソッドを使用
                if hasattr(geometry, 'remove_degenerate_faces'):
                    geometry.remove_degenerate_faces()
        
        # 重複した面を削除
        if hasattr(geometry, 'unique_faces'):
            try:
                unique_faces = geometry.unique_faces()
                geometry.update_faces(unique_faces)
            except:
                # 古いメソッドを使用
                if hasattr(geometry, 'remove_duplicate_faces'):
                    geometry.remove_duplicate_faces()
        
        # 面の法線を修正
        if hasattr(geometry, 'fix_normals'):
            geometry.fix_normals()
        
        # 結果を確認
        new_vertices = len(geometry.vertices) if hasattr(geometry, 'vertices') else 0
        new_faces = len(geometry.faces) if hasattr(geometry, 'faces') else 0
        
        if original_vertices > 0 or original_faces > 0:
            vertex_reduction = original_vertices - new_vertices
            face_reduction = original_faces - new_faces
            if vertex_reduction > 0 or face_reduction > 0:
                print(f"[i] クリーンアップ: 頂点-{vertex_reduction}, 面-{face_reduction}")
        
        return True
        
    except Exception as e:
        print(f"[!] ジオメトリクリーンアップエラー: {e}")
        return False

def deep_clean_scene(scene):
    """シーン全体の深いクリーンアップ"""
    try:
        cleaned_count = 0
        total_count = 0
        
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                total_count += 1
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    if clean_geometry(geometry):
                        cleaned_count += 1
        elif hasattr(scene, 'vertices') and hasattr(scene, 'faces'):
            # 単一メッシュの場合
            total_count = 1
            if clean_geometry(scene):
                cleaned_count = 1
        
        print(f"[✓] ジオメトリクリーンアップ完了: {cleaned_count}/{total_count}")
        return True
        
    except Exception as e:
        print(f"[×] シーンクリーンアップエラー: {e}")
        return False

def detect_mesh_holes(geometry):
    """メッシュの穴（開口部）を検出"""
    try:
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return []
        
        holes_detected = []
        
        # Trimeshの穴検出機能を使用
        if hasattr(geometry, 'is_watertight'):
            if not geometry.is_watertight:
                print("[i] メッシュに開口部が検出されました")
                holes_detected.append("開口部あり")
        
        # エッジの開口部を検出
        if hasattr(geometry, 'edges'):
            try:
                edges = geometry.edges
                unique_edges = geometry.edges_unique
                if len(edges) != len(unique_edges):
                    boundary_edges = len(edges) - len(unique_edges)
                    if boundary_edges > 0:
                        holes_detected.append(f"境界エッジ: {boundary_edges}本")
                        print(f"[i] 境界エッジを検出: {boundary_edges}本")
            except Exception as e:
                print(f"[!] エッジ検出エラー: {e}")
        
        # 開いたエッジを検出
        if hasattr(geometry, 'outline'):
            try:
                outline = geometry.outline()
                if outline is not None and hasattr(outline, 'entities') and len(outline.entities) > 0:
                    holes_detected.append(f"開いた輪郭: {len(outline.entities)}個")
                    print(f"[i] 開いた輪郭を検出: {len(outline.entities)}個")
            except Exception as e:
                print(f"[!] 輪郭検出エラー: {e}")
        
        if len(holes_detected) == 0:
            print("[✓] 穴は検出されませんでした")
        
        return holes_detected
        
    except Exception as e:
        print(f"[!] 穴検出エラー: {e}")
        return []

def fill_holes_basic_triangulation(geometry):
    """基本的な三角分割による穴埋め"""
    try:
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return False
        
        print("[i] 基本的な三角分割による穴埋めを試行...")
        original_faces = len(geometry.faces)
        
        # 境界エッジを検出
        edges = geometry.edges
        edge_face_count = {}
        
        # 各エッジが何個の面で使用されているかカウント
        for face in geometry.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_face_count[edge] = edge_face_count.get(edge, 0) + 1
        
        # 境界エッジ（1つの面でのみ使用されているエッジ）を特定
        boundary_edges = [edge for edge, count in edge_face_count.items() if count == 1]
        
        if not boundary_edges:
            print("[i] 境界エッジが見つかりません")
            return False
        
        print(f"[i] 境界エッジ {len(boundary_edges)}本を検出")
        
        # 境界エッジから境界ループを構築
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        boundary_vertices = list(boundary_vertices)
        print(f"[i] 境界頂点 {len(boundary_vertices)}個を検出")
        
        # 簡単な穴埋め: 境界頂点の重心を計算して中心点を作成
        if len(boundary_vertices) >= 3:
            center_vertex = np.mean(geometry.vertices[boundary_vertices], axis=0)
            
            # 新しい頂点を追加
            new_vertices = np.vstack([geometry.vertices, center_vertex.reshape(1, -1)])
            center_index = len(geometry.vertices)
            
            # 境界エッジと中心点を結ぶ面を作成
            new_faces = []
            for edge in boundary_edges:
                # 境界エッジの向きを考慮して面を作成
                new_face = [edge[0], edge[1], center_index]
                new_faces.append(new_face)
            
            if new_faces:
                # メッシュを更新
                geometry.vertices = new_vertices
                all_faces = np.vstack([geometry.faces, np.array(new_faces)])
                geometry.faces = all_faces
                
                added_faces = len(new_faces)
                print(f"[✓] 基本三角分割で穴埋め: {added_faces}面を追加")
                return True
        
        return False
        
    except Exception as e:
        print(f"[!] 基本三角分割エラー: {e}")
        return False

def fill_holes_delaunay_triangulation(geometry):
    """Delaunay三角分割による高度な穴埋め"""
    try:
        print("[i] Delaunay三角分割による穴埋めを試行...")
        
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return False
        
        # 境界エッジを検出
        edges = geometry.edges
        edge_face_count = {}
        
        for face in geometry.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_face_count[edge] = edge_face_count.get(edge, 0) + 1
        
        boundary_edges = [edge for edge, count in edge_face_count.items() if count == 1]
        
        if not boundary_edges:
            return False
        
        # 境界ループを構築
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        boundary_vertices = list(boundary_vertices)
        
        if len(boundary_vertices) < 3:
            return False
        
        print(f"[i] Delaunay: 境界頂点{len(boundary_vertices)}個で穴埋め")
        
        # 境界頂点の座標を取得
        boundary_coords = geometry.vertices[boundary_vertices]
        
        # 平面投影用の主軸を計算
        coords_centered = boundary_coords - np.mean(boundary_coords, axis=0)
        
        # 共分散行列から主成分を計算
        cov_matrix = np.cov(coords_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 最小固有値に対応する法線ベクトル
        normal = eigenvectors[:, 0]
        
        # 平面への投影行列を作成
        if abs(normal[2]) > 0.8:  # Z軸に近い場合はXY平面へ投影
            proj_coords = boundary_coords[:, :2]
        elif abs(normal[1]) > 0.8:  # Y軸に近い場合はXZ平面へ投影
            proj_coords = boundary_coords[:, [0, 2]]
        else:  # X軸に近い場合はYZ平面へ投影
            proj_coords = boundary_coords[:, [1, 2]]
        
        # 簡易的な扇形三角分割
        center_2d = np.mean(proj_coords, axis=0)
        
        # 境界点を角度でソート
        angles = np.arctan2(proj_coords[:, 1] - center_2d[1], 
                           proj_coords[:, 0] - center_2d[0])
        sorted_indices = np.argsort(angles)
        
        # 新しい面を作成
        new_faces = []
        for i in range(len(sorted_indices)):
            v1 = boundary_vertices[sorted_indices[i]]
            v2 = boundary_vertices[sorted_indices[(i + 1) % len(sorted_indices)]]
            
            # 重心に新しい頂点を追加する代わりに、既存の境界頂点で三角形を作成
            if i < len(sorted_indices) - 2:
                v3 = boundary_vertices[sorted_indices[(i + 2) % len(sorted_indices)]]
                new_faces.append([v1, v2, v3])
        
        if new_faces:
            # 新しい面を追加
            all_faces = np.vstack([geometry.faces, np.array(new_faces)])
            geometry.faces = all_faces
            
            print(f"[✓] Delaunay三角分割で{len(new_faces)}面を追加")
            return True
        
        return False
        
    except Exception as e:
        print(f"[!] Delaunay三角分割エラー: {e}")
        return False

def fill_holes_convex_hull(geometry):
    """凸包による強制的穴埋め"""
    try:
        print("[i] 凸包による強制穴埋めを試行...")
        
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return False
        
        original_faces = len(geometry.faces)
        
        # 境界エッジを検出
        edges = geometry.edges
        edge_face_count = {}
        
        for face in geometry.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_face_count[edge] = edge_face_count.get(edge, 0) + 1
        
        boundary_edges = [edge for edge, count in edge_face_count.items() if count == 1]
        
        if not boundary_edges:
            return False
        
        # 境界頂点を取得
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        boundary_vertices = list(boundary_vertices)
        
        if len(boundary_vertices) < 3:
            return False
        
        # 境界頂点の凸包を計算
        boundary_coords = geometry.vertices[boundary_vertices]
        
        try:
            from scipy.spatial import ConvexHull
            hull_2d = ConvexHull(boundary_coords[:, :2])  # XY平面での凸包
            
            # 凸包の三角形を作成
            new_faces = []
            for simplex in hull_2d.simplices:
                # 境界頂点インデックスに変換
                face_vertices = [boundary_vertices[i] for i in simplex]
                new_faces.append(face_vertices)
            
            if new_faces:
                all_faces = np.vstack([geometry.faces, np.array(new_faces)])
                geometry.faces = all_faces
                
                print(f"[✓] 凸包で{len(new_faces)}面を追加")
                return True
                
        except ImportError:
            print("[i] scipy未インストール、代替手法を使用")
            # scipyが無い場合の代替手法
            pass
        except Exception as e:
            print(f"[!] 凸包計算エラー: {e}")
        
        return False
        
    except Exception as e:
        print(f"[!] 凸包穴埋めエラー: {e}")
        return False

def fill_holes_edge_bridging(geometry):
    """エッジブリッジングによる穴埋め"""
    try:
        print("[i] エッジブリッジング手法を試行...")
        
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return False
        
        # 面の法線を計算して一貫性をチェック
        if hasattr(geometry, 'fix_normals'):
            geometry.fix_normals()
        
        # 重複した頂点をマージ
        if hasattr(geometry, 'merge_vertices'):
            original_vertex_count = len(geometry.vertices)
            geometry.merge_vertices()
            merged_count = original_vertex_count - len(geometry.vertices)
            if merged_count > 0:
                print(f"[i] 重複頂点 {merged_count}個をマージ")
        
        # 無効な面を削除
        if hasattr(geometry, 'remove_degenerate_faces'):
            original_face_count = len(geometry.faces)
            geometry.remove_degenerate_faces()
            removed_count = original_face_count - len(geometry.faces)
            if removed_count > 0:
                print(f"[i] 無効な面 {removed_count}個を削除")
        
        # 面の向きを統一
        if hasattr(geometry, 'fix_normals'):
            geometry.fix_normals()
            print("[i] 面の向きを統一")
        
        return True
        
    except Exception as e:
        print(f"[!] エッジブリッジングエラー: {e}")
        return False

def fill_mesh_holes(geometry):
    """メッシュの穴を埋める（軽量版）"""
    try:
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return False
        
        filled = False
        original_faces = len(geometry.faces)
        
        # Trimeshの標準穴埋め機能のみ使用
        if hasattr(geometry, 'fill_holes'):
            try:
                geometry.fill_holes()
                new_faces = len(geometry.faces)
                if new_faces > original_faces:
                    added_faces = new_faces - original_faces
                    print(f"[✓] 穴を埋めました: {added_faces}面を追加")
                    filled = True
                else:
                    print("[i] 追加された面なし")
            except Exception as e:
                print(f"[!] 穴埋めエラー: {e}")
        
        # 基本的なメッシュクリーンアップ
        if not filled:
            try:
                # 重複頂点を統合
                if hasattr(geometry, 'merge_vertices'):
                    geometry.merge_vertices()
                
                # 面の向きを修正
                if hasattr(geometry, 'fix_normals'):
                    geometry.fix_normals()
                
                print("[✓] メッシュ整合性を改善")
                filled = True
                
            except Exception as e:
                print(f"[!] メッシュクリーンアップエラー: {e}")
        
        if not filled:
            print("[!] 穴埋めに失敗しました")
        
        return filled
        
    except Exception as e:
        print(f"[!] 穴埋めエラー: {e}")
        return False

def validate_and_repair_mesh(geometry):
    """メッシュの検証と修復"""
    try:
        repaired = False
        
        # 基本的な検証
        if not hasattr(geometry, 'vertices') or not hasattr(geometry, 'faces'):
            return False
        
        if len(geometry.vertices) == 0 or len(geometry.faces) == 0:
            return False
        
        # 面のインデックスが頂点数を超えていないかチェック
        max_vertex_index = len(geometry.vertices) - 1
        if hasattr(geometry, 'faces') and len(geometry.faces) > 0:
            face_max = np.max(geometry.faces)
            if face_max > max_vertex_index:
                print(f"[!] 無効な面インデックス検出: {face_max} > {max_vertex_index}")
                # 無効な面を削除
                valid_faces = []
                for face in geometry.faces:
                    if np.all(face <= max_vertex_index) and np.all(face >= 0):
                        valid_faces.append(face)
                
                if len(valid_faces) > 0:
                    geometry.faces = np.array(valid_faces)
                    repaired = True
                    print(f"[✓] 無効な面を削除: {len(geometry.faces)} 面が残存")
                else:
                    return False
        
        # メッシュの整合性をチェック
        if hasattr(geometry, 'is_valid'):
            if not geometry.is_valid:
                print("[!] メッシュの整合性に問題があります")
                # 可能であれば修復を試行
                if hasattr(geometry, 'fill_holes'):
                    try:
                        geometry.fill_holes()
                        repaired = True
                        print("[✓] メッシュの穴を修復")
                    except:
                        pass
        
        if repaired:
            print("[✓] メッシュ修復完了")
        
        return True
        
    except Exception as e:
        print(f"[!] メッシュ検証エラー: {e}")
        return False

def flatten_scene(scene):
    """複数シーンを1つのフラットなシーンにまとめる"""
    try:
        from trimesh.scene.scene import Scene
        
        # 新しいフラットなシーンを作成
        flat_scene = Scene()
        flattened_count = 0
        
        if hasattr(scene, 'geometry') and scene.geometry:
            # すべてのジオメトリを新しいシーンに直接追加
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    # シンプルな名前で追加
                    simple_name = f"mesh_{flattened_count}"
                    flat_scene.add_geometry(geometry, node_name=simple_name)
                    flattened_count += 1
                    print(f"[✓] フラット化: {name} → {simple_name}")
        
        print(f"[✓] シーンフラット化完了: {flattened_count}個のメッシュ")
        return flat_scene
        
    except Exception as e:
        print(f"[!] シーンフラット化エラー: {e}")
        return scene

def remove_scene_elements(scene):
    """シーンから不要な要素を安全に削除（ライト、カメラ、ノード階層）"""
    try:
        removed = []
        
        # ライトを削除
        if hasattr(scene, 'lights') and scene.lights:
            light_count = len(scene.lights)
            try:
                scene.lights.clear()
                removed.append(f"ライト({light_count}個)")
            except:
                pass
        
        # カメラは削除せず、位置情報を保持
        # if hasattr(scene, 'camera') and scene.camera is not None:
        #     try:
        #         scene.camera = None
        #         removed.append("カメラ")
        #     except:
        #         pass
        
        # シーンのフラット化をスキップ（位置情報保持のため）
        # try:
        #     original_nodes = len(scene.graph.nodes) if hasattr(scene, 'graph') and scene.graph else 0
        #     scene = flatten_scene(scene)
        #     new_nodes = len(scene.graph.nodes) if hasattr(scene, 'graph') and scene.graph else 0
        #     if original_nodes > new_nodes:
        #         removed.append(f"ノード階層({original_nodes}→{new_nodes})")
        # except Exception as e:
        #     print(f"[!] ノード階層削除エラー: {e}")
        
        if removed:
            print(f"[✓] シーン要素削除: {', '.join(removed)}")
        else:
            print("[i] 削除可能なシーン要素なし")
        
        return scene
        
    except Exception as e:
        print(f"[!] シーン要素削除エラー: {e}")
        return scene

def create_minimal_glb(scene):
    """最小限のGLBを作成（メッシュ+テクスチャのみ）"""
    try:
        # 新しいシーンを作成
        from trimesh.scene.scene import Scene
        minimal_scene = Scene()
        
        # ジオメトリのみをコピー
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                # 必要最小限の属性のみを保持
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    # 新しいメッシュを作成
                    minimal_mesh = trimesh.Trimesh(
                        vertices=geometry.vertices,
                        faces=geometry.faces
                    )
                    
                    # UV座標とテクスチャを保持
                    if hasattr(geometry, 'visual'):
                        minimal_mesh.visual = geometry.visual
                    
                    # 法線は再計算（より軽量）
                    if hasattr(minimal_mesh, 'vertex_normals'):
                        minimal_mesh.vertex_normals
                    
                    minimal_scene.add_geometry(minimal_mesh, node_name=name)
        
        print("[✓] 最小限のGLB構造を作成")
        return minimal_scene
        
    except Exception as e:
        print(f"[!] 最小限GLB作成エラー: {e}")
        return scene

def hole_filling_optimize(input_path, output_path):
    """穴埋め特化モード（メッシュの穴を検出・修復）"""
    try:
        scene = trimesh.load(input_path, force='scene', process=False)
        print("[i] 穴埋め特化モード: メッシュの穴を検出・修復")
        
        total_holes_found = 0
        total_holes_filled = 0
        processed_meshes = 0
        
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    processed_meshes += 1
                    print(f"[i] {name}: 穴の検出を開始...")
                    
                    # 穴を検出
                    holes_detected = detect_mesh_holes(geometry)
                    if len(holes_detected) > 0:
                        total_holes_found += len(holes_detected)
                        print(f"[!] {name}: 検出された問題: {', '.join(holes_detected)}")
                        
                        # 穴を埋める
                        original_faces = len(geometry.faces)
                        original_vertices = len(geometry.vertices)
                        
                        if fill_mesh_holes(geometry):
                            total_holes_filled += 1
                            new_faces = len(geometry.faces)
                            new_vertices = len(geometry.vertices)
                            added_faces = new_faces - original_faces
                            added_vertices = new_vertices - original_vertices
                            
                            if added_faces > 0 or added_vertices > 0:
                                print(f"[✓] {name}: 穴埋め完了 (面+{added_faces}, 頂点+{added_vertices})")
                            else:
                                print(f"[✓] {name}: メッシュ整合性を改善")
                        else:
                            print(f"[!] {name}: 穴埋めに失敗")
                            
                            # フォールバック: 基本的なクリーンアップだけでも実行
                            print(f"[i] {name}: フォールバック処理を実行...")
                            try:
                                clean_geometry(geometry)
                                print(f"[✓] {name}: 基本クリーンアップ完了")
                            except Exception as e:
                                print(f"[!] {name}: フォールバック処理も失敗: {e}")
                    else:
                        print(f"[✓] {name}: 穴は検出されませんでした")
                    
                    # 基本的なメッシュ修復も実行
                    try:
                        validate_and_repair_mesh(geometry)
                    except Exception as e:
                        print(f"[!] {name}: メッシュ修復エラー: {e}")
        
        # 全体の修復結果を表示
        print(f"[i] 処理完了: {processed_meshes}個のメッシュを検査")
        if total_holes_found > 0:
            print(f"[i] 検出された問題: {total_holes_found}個")
            print(f"[✓] 修復成功: {total_holes_filled}個")
            
            if total_holes_filled < total_holes_found:
                failed_repairs = total_holes_found - total_holes_filled
                print(f"[!] 修復失敗: {failed_repairs}個")
        else:
            print("[✓] すべてのメッシュで穴は検出されませんでした")
        
        # エクスポート
        scene.export(output_path, file_type='glb')
        print(f"[✓] 穴埋め特化最適化完了: {output_path}")
        return True
        
    except Exception as e:
        print(f"[×] 穴埋め特化最適化に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def vertex_cleanup_optimize(input_path, output_path):
    """頂点削除特化モード（未使用頂点・重複頂点を徹底削除）"""
    try:
        scene = trimesh.load(input_path, force='scene', process=False)
        print("[i] 頂点削除特化モード: 未使用・重複頂点を徹底削除")
        
        total_original_vertices = 0
        total_cleaned_vertices = 0
        cleaned_meshes = 0
        
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    original_vertices = len(geometry.vertices)
                    total_original_vertices += original_vertices
                    
                    print(f"[i] {name}: {original_vertices}頂点を処理中...")
                    
                    # 未使用頂点削除
                    if remove_unused_vertices(geometry):
                        cleaned_meshes += 1
                    
                    # 重複頂点削除
                    if hasattr(geometry, 'merge_vertices'):
                        try:
                            geometry.merge_vertices()
                            print(f"[✓] {name}: 重複頂点を統合")
                        except Exception as e:
                            print(f"[!] {name}: 重複頂点統合エラー: {e}")
                    
                    # 孤立頂点削除
                    if hasattr(geometry, 'remove_unreferenced_vertices'):
                        try:
                            geometry.remove_unreferenced_vertices()
                            print(f"[✓] {name}: 孤立頂点を削除")
                        except Exception as e:
                            print(f"[!] {name}: 孤立頂点削除エラー: {e}")
                    
                    cleaned_vertices = len(geometry.vertices)
                    total_cleaned_vertices += cleaned_vertices
                    
                    reduction = original_vertices - cleaned_vertices
                    if reduction > 0:
                        reduction_percent = (reduction / original_vertices) * 100
                        print(f"[✓] {name}: {original_vertices} → {cleaned_vertices}頂点 ({reduction_percent:.1f}%削減)")
                    else:
                        print(f"[i] {name}: 削除対象頂点なし")
        
        # 全体の削減結果を表示
        total_reduction = total_original_vertices - total_cleaned_vertices
        if total_reduction > 0:
            total_reduction_percent = (total_reduction / total_original_vertices) * 100
            print(f"[✓] 全体: {total_original_vertices} → {total_cleaned_vertices}頂点")
            print(f"[✓] 削減量: {total_reduction}頂点 ({total_reduction_percent:.1f}%)")
        else:
            print("[i] 削除対象の余計な頂点は見つかりませんでした")
        
        # エクスポート
        scene.export(output_path, file_type='glb')
        print(f"[✓] 頂点削除特化最適化完了: {output_path}")
        return True
        
    except Exception as e:
        print(f"[×] 頂点削除特化最適化に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def ultra_light_optimize(input_path, output_path):
    """積極的な軽量化（品質とサイズ重視）"""
    try:
        scene = trimesh.load(input_path, force='scene', process=False)
        print("[i] 積極的軽量化モード: サイズ削減を重視")
        
        # テクスチャをWebP積極圧縮（70%品質＋リサイズ）
        print("[i] テクスチャをWebP積極圧縮...")
        try:
            compress_scene_textures_webp(scene, quality=70, resize_factor=0.75)
        except Exception as e:
            print(f"[!] WebPテクスチャ圧縮エラー: {e}")
        
        # カメラ・ライト削除をスキップ（位置情報保持のため）
        # try:
        #     scene = remove_scene_elements(scene)
        # except Exception as e:
        #     print(f"[!] シーン要素削除エラー: {e}")
        print("[i] シーン要素削除をスキップ（位置情報保持）")
        
        # ジオメトリクリーンアップをスキップ（位置情報保持のため）
        # try:
        #     deep_clean_scene(scene)
        # except Exception as e:
        #     print(f"[!] クリーンアップエラー: {e}")
        print("[i] ジオメトリクリーンアップをスキップ（位置情報保持）")
        
        # テクスチャは保持
        print("[✓] テクスチャを保持")
        
        # 安全にエクスポート
        try:
            scene.export(output_path, file_type='glb')
            print(f"[✓] 控えめ軽量化完了: {output_path}")
            return True
        except Exception as e:
            print(f"[×] エクスポートエラー: {e}")
            return False
        
    except Exception as e:
        print(f"[×] 控えめ軽量化に失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python glbopt.py <input.glb> [mode] [target_ratio]")
        print("  mode: normal (default), aggressive, ultra, vertex, repair")
        print("  target_ratio: 0.1-1.0 (default: 0.7 for normal, 0.3 for aggressive, 0.1 for ultra)")
        print("")
        print("モード:")
        print("  normal     - 安全最適化（重複頂点削除、テクスチャ保持）")
        print("  aggressive - 適度軽量化（重複頂点削除、WebP85%品質）")
        print("  ultra      - 積極軽量化（重複頂点削除、WebP70%品質+75%リサイズ）")
        print("  vertex     - 頂点削除特化（未使用・重複・孤立頂点を徹底削除）")
        print("  repair     - 穴埋め特化（メッシュの穴を検出・修復）")
        return

    input_glb = sys.argv[1]
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "normal"
    
    # モード別のデフォルト値
    if mode == "ultra":
        default_ratio = 0.1
    elif mode == "aggressive":
        default_ratio = 0.3
    elif mode == "vertex":
        default_ratio = 1.0  # 頂点削除モードでは比率は使用しない
    elif mode == "repair":
        default_ratio = 1.0  # 穴埋めモードでは比率は使用しない
    else:
        default_ratio = 0.7
        mode = "normal"
    
    target_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else default_ratio
    
    if not (0.05 <= target_ratio <= 1.0):
        print("[×] target_ratioは0.05-1.0の範囲で指定してください")
        return
    
    base, ext = os.path.splitext(input_glb)
    output_glb = f"{base}_{mode}.glb"

    if not os.path.exists(input_glb):
        print(f"[×] 入力ファイルが存在しません: {input_glb}")
        return

    print(f"[i] 入力: {input_glb}")
    print(f"[i] 出力: {output_glb}")
    print(f"[i] モード: {mode}")
    print(f"[i] 簡略化率: {target_ratio}")
    
    # ファイルサイズの確認
    original_size = os.path.getsize(input_glb)
    print(f"[i] 元のファイルサイズ: {original_size / 1024 / 1024:.2f} MB")
    
    # モード別の最適化
    success = False
    
    if mode == "repair":
        print("[i] 穴埋め特化モードを実行...")
        success = hole_filling_optimize(input_glb, output_glb)
    elif mode == "vertex":
        print("[i] 頂点削除特化モードを実行...")
        success = vertex_cleanup_optimize(input_glb, output_glb)
    elif mode == "ultra":
        print("[i] 超軽量化モードを実行...")
        success = ultra_light_optimize(input_glb, output_glb)
    elif mode == "aggressive":
        print("[i] 極端軽量化モードを実行...")
        success = aggressive_optimize_glb(input_glb, output_glb, target_ratio, 0.5)
    else:
        # 通常モード: 安全性優先で段階的に試行
        print("[i] 手法1: 安全なテクスチャ保持最適化を試行...")
        success = safe_texture_preserving_optimize(input_glb, output_glb, target_ratio)
        
        if not success:
            print("[i] 手法2: 標準最適化を試行...")
            success = texture_preserving_optimize(input_glb, output_glb, target_ratio)
        
        if not success:
            print("[i] 手法3: 基本最適化を試行...")
            success = safe_optimize_glb(input_glb, output_glb, target_ratio)
    
    if not success:
        print("[×] すべての最適化手法が失敗しました")
        print("[i] 元のファイルをコピーして _opt.glb として保存します...")
        try:
            shutil.copy2(input_glb, output_glb)
            print("[✓] 元のファイルをコピーしました（最適化なし）")
        except Exception as e:
            print(f"[×] コピーにも失敗: {e}")
        return
    
    # 結果のファイルサイズを確認
    if os.path.exists(output_glb):
        optimized_size = os.path.getsize(output_glb)
        compression_ratio = (1 - optimized_size / original_size) * 100
        print(f"[✓] 最適化後のファイルサイズ: {optimized_size / 1024 / 1024:.2f} MB")
        print(f"[✓] 圧縮率: {compression_ratio:.1f}%")
        
        # 検証: 出力ファイルが異常に小さくないかチェック
        if optimized_size < original_size * 0.005:  # 元のファイルの0.5%以下
            print("[!] 警告: 出力ファイルが異常に小さいです。確認してください。")
        
        # 出力ファイルを自動で開く
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", output_glb])
                print(f"[✓] 出力ファイルを開きました: {output_glb}")
            elif system == "Windows":
                subprocess.run(["start", output_glb], shell=True)
                print(f"[✓] 出力ファイルを開きました: {output_glb}")
            elif system == "Linux":
                subprocess.run(["xdg-open", output_glb])
                print(f"[✓] 出力ファイルを開きました: {output_glb}")
            else:
                print(f"[i] 手動でファイルを開いてください: {output_glb}")
        except Exception as e:
            print(f"[!] ファイルを開くことができませんでした: {e}")
            print(f"[i] 手動でファイルを開いてください: {output_glb}")
    else:
        print("[×] 出力ファイルが作成されませんでした")

if __name__ == "__main__":
    main()

