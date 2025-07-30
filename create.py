import sys 
from PIL import Image
import numpy as np
import trimesh
from trimesh.exchange import gltf
import os

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("使用方法: python createcoin.py <画像ファイル> [形状]")
    print("形状: lens (デフォルト), coin, cube, ball, tray, cap")
    print()
    print("利用例:")
    print("  python createcoin.py image.png        # 凸レンズ形状")
    print("  python createcoin.py image.png coin   # コイン形状")
    print("  python createcoin.py image.png cube   # キューブ形状")
    print("  python createcoin.py image.png ball   # ボール形状")
    print("  python createcoin.py image.png lens   # 凸レンズ形状")
    print("  python createcoin.py image.png tray   # トレイ形状（コインを90度倒した形）")
    print("  python createcoin.py image.png cap  # レンズトレイ形状（レンズを90度倒した形）")
    sys.exit(1)

input_file = sys.argv[1]
shape = sys.argv[2] if len(sys.argv) == 3 else "lens"

if shape not in ["coin", "cube", "ball", "lens", "tray", "cap"]:
    print("エラー: 有効な形状は coin, cube, ball, lens, tray, cap です")
    sys.exit(1)

base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = f"{base_name}_{shape}.glb"

try:
    image = Image.open(input_file).convert("RGBA")
except FileNotFoundError:
    print(f"エラー: 画像ファイル '{input_file}' が見つかりません")
    sys.exit(1)
except Exception as e:
    print(f"エラー: 画像ファイルを読み込めません - {e}")
    sys.exit(1)

def create_coin():
    radius = 1.0 
    thickness = 0.1 
    
    coin = trimesh.creation.cylinder(radius=radius, height=thickness, sections=128)
    
    bevel_radius = 0.02
    vertices = coin.vertices.copy()
    for i, vertex in enumerate(vertices):
        x, y, z = vertex
        if abs(z - thickness/2) < 1e-6 or abs(z + thickness/2) < 1e-6:
            r = np.sqrt(x*x + y*y)
            if r > radius - bevel_radius:
                factor = (radius - bevel_radius) / r
                vertices[i][0] *= factor
                vertices[i][1] *= factor
                z_offset = bevel_radius * (1 - factor)
                if z > 0:
                    vertices[i][2] -= z_offset
                else:
                    vertices[i][2] += z_offset
    
    coin.vertices = vertices
    coin.vertex_normals
    coin.face_normals
    
    uvs = np.array([
        [0.5 + 0.5 * (v[0] / radius), 0.5 + 0.5 * (v[1] / radius)] for v in coin.vertices
    ])
    return coin, uvs

def create_cube():
    size = 2.0
    cube = trimesh.creation.box([size, size, size])
    cube.vertex_normals
    cube.face_normals
    
    uvs = np.array([
        [0.5 + 1.0 * (v[0] / size), 0.5 + 1.0 * (v[1] / size)] for v in cube.vertices
    ])
    return cube, uvs

def create_ball():
    radius = 1.0
    ball = trimesh.creation.icosphere(subdivisions=5, radius=radius)
    ball.vertex_normals
    ball.face_normals
    
    uvs = np.array([
        [0.5 + 0.5 * (v[0] / radius), 0.5 + 0.5 * (v[1] / radius)] for v in ball.vertices
    ])
    return ball, uvs

def create_lens():
    radius = 1.0
    height = 0.1
    
    # 手動で凸レンズを作成
    theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
    r_values = np.linspace(0, radius, 16)
    
    vertices = []
    faces = []
    
    # 上面（凸面）
    for i, r in enumerate(r_values):
        for j, t in enumerate(theta):
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = height/2 * np.sqrt(max(0, 1 - (r/radius)**2))
            vertices.append([x, y, z])
    
    # 下面（凸面）
    for i, r in enumerate(r_values):
        for j, t in enumerate(theta):
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = -height/2 * np.sqrt(max(0, 1 - (r/radius)**2))
            vertices.append([x, y, z])
    
    # 面を作成
    num_theta = len(theta)
    num_r = len(r_values)
    
    # 上面の面
    for i in range(num_r - 1):
        for j in range(num_theta):
            v1 = i * num_theta + j
            v2 = i * num_theta + (j + 1) % num_theta
            v3 = (i + 1) * num_theta + j
            v4 = (i + 1) * num_theta + (j + 1) % num_theta
            
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])
    
    # 下面の面
    offset = num_r * num_theta
    for i in range(num_r - 1):
        for j in range(num_theta):
            v1 = offset + i * num_theta + j
            v2 = offset + i * num_theta + (j + 1) % num_theta
            v3 = offset + (i + 1) * num_theta + j
            v4 = offset + (i + 1) * num_theta + (j + 1) % num_theta
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    # 側面の接続
    for j in range(num_theta):
        v1 = (num_r - 1) * num_theta + j
        v2 = (num_r - 1) * num_theta + (j + 1) % num_theta
        v3 = offset + (num_r - 1) * num_theta + j
        v4 = offset + (num_r - 1) * num_theta + (j + 1) % num_theta
        
        faces.append([v1, v3, v2])
        faces.append([v2, v4, v3])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    lens = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    lens.vertex_normals
    lens.face_normals
    
    uvs = np.array([
        [0.5 + 0.5 * (v[0] / radius), 0.5 + 0.5 * (v[1] / radius)] for v in lens.vertices
    ])
    return lens, uvs

def create_tray():
    radius = 1.0 
    thickness = 0.1 
    
    # コインと同じ形状を作成
    coin = trimesh.creation.cylinder(radius=radius, height=thickness, sections=128)
    
    bevel_radius = 0.02
    vertices = coin.vertices.copy()
    for i, vertex in enumerate(vertices):
        x, y, z = vertex
        if abs(z - thickness/2) < 1e-6 or abs(z + thickness/2) < 1e-6:
            r = np.sqrt(x*x + y*y)
            if r > radius - bevel_radius:
                factor = (radius - bevel_radius) / r
                vertices[i][0] *= factor
                vertices[i][1] *= factor
                z_offset = bevel_radius * (1 - factor)
                if z > 0:
                    vertices[i][2] -= z_offset
                else:
                    vertices[i][2] += z_offset
    
    coin.vertices = vertices
    coin.vertex_normals
    coin.face_normals
    
    # 90度回転（X軸周りに回転してコインを前に倒す）
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    coin.apply_transform(rotation_matrix)
    
    # 真下から見た時に正face になるようにUVマッピング（X-Z平面を使用）
    uvs = np.array([
        [0.5 + 0.5 * (v[0] / radius), 0.5 + 0.5 * (v[2] / radius)] for v in coin.vertices
    ])
    return coin, uvs

def create_cap():
    radius = 1.0
    height = 0.1
    
    # レンズと同じ形状を作成
    theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
    r_values = np.linspace(0, radius, 16)
    
    vertices = []
    faces = []
    
    # 上面（凸面）
    for i, r in enumerate(r_values):
        for j, t in enumerate(theta):
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = height/2 * np.sqrt(max(0, 1 - (r/radius)**2))
            vertices.append([x, y, z])
    
    # 下面（凸面）
    for i, r in enumerate(r_values):
        for j, t in enumerate(theta):
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = -height/2 * np.sqrt(max(0, 1 - (r/radius)**2))
            vertices.append([x, y, z])
    
    # 面を作成
    num_theta = len(theta)
    num_r = len(r_values)
    
    # 上面の面
    for i in range(num_r - 1):
        for j in range(num_theta):
            v1 = i * num_theta + j
            v2 = i * num_theta + (j + 1) % num_theta
            v3 = (i + 1) * num_theta + j
            v4 = (i + 1) * num_theta + (j + 1) % num_theta
            
            faces.append([v1, v3, v2])
            faces.append([v2, v3, v4])
    
    # 下面の面
    offset = num_r * num_theta
    for i in range(num_r - 1):
        for j in range(num_theta):
            v1 = offset + i * num_theta + j
            v2 = offset + i * num_theta + (j + 1) % num_theta
            v3 = offset + (i + 1) * num_theta + j
            v4 = offset + (i + 1) * num_theta + (j + 1) % num_theta
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    # 側面の接続
    for j in range(num_theta):
        v1 = (num_r - 1) * num_theta + j
        v2 = (num_r - 1) * num_theta + (j + 1) % num_theta
        v3 = offset + (num_r - 1) * num_theta + j
        v4 = offset + (num_r - 1) * num_theta + (j + 1) % num_theta
        
        faces.append([v1, v3, v2])
        faces.append([v2, v4, v3])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    lens = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    lens.vertex_normals
    lens.face_normals
    
    # 90度回転（X軸周りに回転してレンズを前に倒す）
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    lens.apply_transform(rotation_matrix)
    
    # 真下から見た時に正面になるようにUVマッピング（X-Z平面を使用）
    uvs = np.array([
        [0.5 + 0.5 * (v[0] / radius), 0.5 + 0.5 * (v[2] / radius)] for v in lens.vertices
    ])
    return lens, uvs

if shape == "coin":
    mesh, uvs = create_coin()
elif shape == "cube":
    mesh, uvs = create_cube()
elif shape == "ball":
    mesh, uvs = create_ball()
elif shape == "lens":
    mesh, uvs = create_lens()
elif shape == "tray":
    mesh, uvs = create_tray()
elif shape == "cap":
    mesh, uvs = create_cap()

material = trimesh.visual.material.PBRMaterial(
    baseColorTexture=image,
    baseColorFactor=[1.2, 1.2, 1.2, 1.0],  # 色を強調
    metallicFactor=0.0,  # メタリック感を無効
    roughnessFactor=0.6,  # 滑らかにして反射を増加
    emissiveFactor=[0.0, 0.0, 0.0]  # 自己発光を無効化
)

mesh.visual = trimesh.visual.texture.TextureVisuals(
    uv=uvs, 
    material=material
)

with open(output_file, "wb") as f:
    f.write(gltf.export_glb(mesh))

print(f"GLBファイルを生成しました: {output_file}")

# ファイルを開く
try:
    os.system(f"open '{output_file}'")
except Exception as e:
    print(f"ファイルを開けませんでした: {e}")

