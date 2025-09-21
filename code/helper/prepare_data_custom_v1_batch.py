import os, io
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import glob
import h5py
import numpy as np
import torch
from PIL import Image
from openexr_numpy import imread
import json, shutil
from tqdm import tqdm
from pathlib import Path
import trimesh
import open3d as o3d
import objaverse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import tempfile

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from helper.network import DinoWrapper
from helper.feat_pc_modules import fuse_feature_rgbd, vis_pca, fuse_feature_rgbd_batch
import objaverse.xl as oxl
from pathlib import Path
import trimesh

# for dino feature 3D lifting


def get_intri(target_im):
    h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = torch.tensor([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)

    return K.to(torch.float32)


def normalize_to_unit_cube(meshes: torch.Tensor):
    min_xyz = meshes.min(dim=0).values
    max_xyz = meshes.max(dim=0).values

    # Compute center and scale
    center = (min_xyz + max_xyz) / 2
    scale = (max_xyz - min_xyz).max()

    # Normalize to [-0.5, 0.5]
    normalized_meshes = (meshes - center) / scale

    return normalized_meshes


def trimesh_to_pytorch3d(mesh_raw):
    all_verts = []
    all_faces = []
    vert_offset = 0

    # Process each mesh in the scene
    for mesh_name, mesh in mesh_raw.geometry.items():
        # Get vertices and faces
        verts = torch.tensor(mesh.vertices.astype(np.float32))
        faces = torch.tensor(mesh.faces.astype(np.int64))
        
        # Append to lists
        all_verts.append(verts)
        all_faces.append(faces + vert_offset)  # Offset face indices
        
        # Update vertex offset for next mesh
        vert_offset += verts.shape[0]

    # If the scene has only one mesh, use it directly
    if len(all_verts) == 1:
        verts_tensor = all_verts[0]
        faces_tensor = all_faces[0]
    # Otherwise concatenate all meshes
    else:
        verts_tensor = torch.cat(all_verts, dim=0)
        faces_tensor = torch.cat(all_faces, dim=0)

    # Create a batch of meshes (with batch size 1)
    verts_batch = verts_tensor.unsqueeze(0)  # [1, N, 3]
    faces_batch = faces_tensor.unsqueeze(0)  # [1, F, 3]

    meshes = Meshes(verts=verts_batch, faces=faces_batch)

    return meshes


def get_model(id_info, model_id):
    uid_name = id_info[model_id].split('/')[1]
    uid = [Path(uid_name).stem]

    objects = objaverse.load_objects(uids=uid, download_processes=1)

    mesh_raw = trimesh.load(list(objects.values())[0])

    meshes = trimesh_to_pytorch3d(mesh_raw)
    sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
    mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=4096)
    mesh_pc = mesh_pc.squeeze(0)
    # mesh_pc = mesh_pc.to(torch.float32)
    sampled_points = normalize_to_unit_cube(mesh_pc)

    return sampled_points


# def get_model(id_info, model_id):
#     # Extract the unique identifier for the model
#     uid_name = id_info[model_id].split('/')[1]
#     uid = Path(uid_name).stem

#     # Retrieve annotations for the specified UID
#     annotations = oxl.get_annotations()
#     model_annotation = annotations[annotations['uid'] == uid]

#     if model_annotation.empty:
#         raise ValueError(f"No annotation found for UID: {uid}")

#     # Download the object corresponding to the UID
#     oxl.download_objects(objects=model_annotation)

#     # Construct the path to the downloaded model file
#     download_dir = oxl.DEFAULT_DOWNLOAD_DIR
#     model_path = Path(download_dir) / uid / 'model.obj'  # Adjust the filename and extension as needed

#     if not model_path.exists():
#         raise FileNotFoundError(f"Model file not found at: {model_path}")

#     # Load the mesh using trimesh
#     mesh_raw = trimesh.load(model_path)

#     # Convert the mesh to PyTorch3D format
#     meshes = trimesh_to_pytorch3d(mesh_raw)

#     # Sample points from the mesh
#     sample_mesh_pc, sample_pc_norm = sample_points_from_meshes(meshes, return_normals=True)
#     mesh_pc, sample_idx = sample_farthest_points(sample_mesh_pc, K=4096)
#     mesh_pc = mesh_pc.squeeze(0)

#     # Normalize the sampled points to fit within a unit cube
#     sampled_points = normalize_to_unit_cube(mesh_pc)

#     return sampled_points

def process_single_subfolder(subfolder_info):
    """
    Process a single subfolder/view
    
    Args:
        subfolder_info: tuple containing (subfolder, object_path, object_name, get_intri, imread)
    
    Returns:
        dict with processed data or None if processing failed
    """
    subfolder, object_path, object_name, get_intri, imread = subfolder_info
    
    subfolder_path = os.path.join(object_path, subfolder)
    
    # Get file paths
    json_file = os.path.join(subfolder_path, f"{subfolder}.json")
    image_file = os.path.join(subfolder_path, f"{subfolder}.png")
    normal_file = os.path.join(subfolder_path, f"{subfolder}_nd.exr")

    # print("show image_file:", image_file)
    
    # Check if all required files are there
    if not (os.path.exists(json_file) and os.path.exists(image_file) and os.path.exists(normal_file)):
        return None
    if cv2.imread(image_file) is None:
        return None

    try:
        # Load camera poses from JSON
        pose = {}
        c2w = np.eye(4).astype('float32')
        
        with open(json_file, 'r') as file:
            json_content = json.load(file)
        
        # Extract camera-to-world matrix as specified
        c2w[:3, 0] = np.array(json_content['x'], dtype=np.float32)
        c2w[:3, 1] = np.array(json_content['y'], dtype=np.float32)
        c2w[:3, 2] = np.array(json_content['z'], dtype=np.float32)
        c2w[:3, 3] = np.array(json_content['origin'], dtype=np.float32)
        
        # Extract FOV and bbox if available
        pose['fov'] = np.array([json_content['x_fov'], json_content['y_fov']], dtype=np.float32)
        pose['bbox'] = np.array(json_content['bbox'], dtype=np.float32)
        pose['c2w'] = c2w
        
        # Load RGB image
        img = cv2.imread(image_file)
        img_resized = cv2.resize(img, (420, 420))

        # Get intrinsic
        cam_intri = get_intri(img_resized)
        pose['cam_k'] = cam_intri.detach().cpu().numpy()
        
        # Load normal map
        try:
            normald = imread(normal_file)
            normal = normald[..., :3]
            normal_norm = np.linalg.norm(normal, 2, axis=-1, keepdims=True)
            normal = normal / np.maximum(normal_norm, 1e-10)  # Avoid division by zero
            normal = normal[..., [2, 0, 1]]
            normal[..., [0, 1]] = -normal[..., [0, 1]]
            normal = ((normal + 1) / 2 * 255).astype('uint8')

            depth = normald[:, :, -1]
            depth[np.where(depth < 0.5)] = 0
            depth_resized = cv2.resize(depth, (420, 420), interpolation=cv2.INTER_NEAREST)
            normal_resized = cv2.resize(normal, (420, 420))

            # Get mask
            mask = np.zeros_like(depth_resized, dtype=np.uint8)
            mask[np.where(depth_resized != 0)] = 1

            # Apply erosion to mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)

            depth_resized = depth_resized * mask

            return {
                'subfolder': subfolder,
                'normal': normal_resized,
                'depth': depth_resized,
                'image': img_resized,
                'mask': mask[:, :, None],
                'pose': pose,
                'camera': (cam_intri, torch.from_numpy(c2w))
            }
            
        except Exception as e:
            print(f"Warning: Could not load normal map for view {subfolder} of {object_name}: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Error processing view {subfolder} of {object_name}: {str(e)}")
        return None
    

def process_subfolders_parallel(subfolders, object_path, object_name, get_intri, imread, max_workers=None, method='process'):
    """
    Process subfolders in parallel
    
    Args:
        subfolders: list of subfolder names
        object_path: path to the object directory
        object_name: name of the object
        get_intri: function to get camera intrinsics
        imread: function to read images
        max_workers: maximum number of workers (None for auto)
        method: 'process' for ProcessPoolExecutor, 'thread' for ThreadPoolExecutor
    
    Returns:
        tuple of (normals, depths, images, masks, poses, cameras) lists
    """
    if max_workers is None:
        max_workers = min(len(subfolders), mp.cpu_count())
    
    # Prepare data for each worker
    subfolder_infos = [
        (subfolder, object_path, object_name, get_intri, imread) 
        for subfolder in subfolders
    ]
    
    # Choose executor based on method
    if method == 'process':
        ExecutorClass = ProcessPoolExecutor
    else:
        ExecutorClass = ThreadPoolExecutor
        max_workers = min(max_workers, 8)  # Limit threads for I/O bound tasks
    
    results = []
    
    with ExecutorClass(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_subfolder = {
            executor.submit(process_single_subfolder, info): info[0] 
            for info in subfolder_infos
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_subfolder):
            subfolder = future_to_subfolder[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {subfolder}: {str(e)}")
    
    # Sort results by subfolder name to maintain order
    results.sort(key=lambda x: x['subfolder'])
    
    # Extract data into separate lists
    normals = []
    depths = []
    images = []
    masks = []
    poses = []
    cameras = []
    
    for result in results:
        normals.append(result['normal'])
        depths.append(result['depth'])
        images.append(result['image'])
        masks.append(result['mask'])
        poses.append(result['pose'])
        cameras.append(result['camera'])
    
    return normals, depths, images, masks, poses, cameras


def process_category(category_path, output_path, feature_model_path, feature_vis_path,id_info):
    """
    Process all objects in a category and create a single h5py file.
    """
    # # Create output directory if it doesn't exist
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get all object folders in this category
    category_name = os.path.basename(category_path)
    object_folders = sorted(glob.glob(category_path+'/*/*'))
    # object_folders = [f for f in os.listdir(sub_folders) if os.path.isdir(os.path.join(category_path, f))]

    category_data = {}
    
    # # Create temporary directory for extraction
    # os.makedirs('temp', exist_ok=True)
    backbone = DinoWrapper(model_name='dinov2_vits14', is_train=False, ).to('cuda').eval()
    # Create h5py file for this category
    for i, obj_folder in enumerate(tqdm(object_folders, desc=f"Processing {category_name}")):


        # clear_cuda_memory()

        parts = obj_folder.strip("/").split("/")
        shape_name_save = f"{parts[-2]}/{parts[-1]}"


        print("show output_path:",output_path)
        print("show shape_name_save:",shape_name_save)
      


        object_name =f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.


        # safe_filename = 
        # print("show shape_name_save:",safe_filename)# 1000/5003641
        h5_file_path = os.path.join(output_path, f"{shape_name_save.replace('/', '_')}.h5")

        # h5_file_path = os.path.join(output_path, f"{shape_name_save}.h5")
        # print("show h5_file_path:",h5_file_path)
        if os.path.exists(h5_file_path):


            with h5py.File(h5_file_path, 'r') as h5_file:

                    print(list(h5_file.keys())) # ['armor_0']
                    bad_key = list(h5_file.keys())[0]
                    save_key = shape_name_save.replace('/', '_')

                    if bad_key==save_key:
                        print("skip")
                        continue    


            # Now rename the key by creating a new file
            target_dir = os.path.dirname(h5_file_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5', dir=target_dir) as tmp:
                tmp_path = tmp.name
            
            try:
                # Copy data with new key name
                with h5py.File(h5_file_path, 'r') as src:
                    with h5py.File(tmp_path, 'w') as dst:
                        # Copy the dataset with new name
                        src.copy(bad_key, dst, name=save_key)
                        
                        # Copy file-level attributes if any
                        for attr_key, attr_value in src.attrs.items():
                            dst.attrs[attr_key] = attr_value
                
                # Replace original file
                os.replace(tmp_path, h5_file_path)
                print(f"Successfully renamed '{bad_key}' to '{save_key}'")
                
            except Exception as e:
                # Clean up temp file if error occurs
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                print(f"Error renaming key: {e}")
                raise e
                    


            # print("should have: shape_name_save.replace('/', '_'):",save_key)# 1000_5003641
            # print("bad_key:",bad_key)
            # exit(0)


            continue


        object_path = obj_folder
        if object_path.endswith('.tar.gz'):
            continue
        # print("show object_path object_path:",object_path)

        # exit(0)
        object_id = "/".join(object_path.split("/")[-2:])

        # Get all subfolders in this object folder (00000, 00001, ..., 00039)
        candidates = [d for d in os.listdir(object_path) if d.startswith("campos_512_") and os.path.isdir(os.path.join(object_path, d))]

        # Use the first match (sorted for consistency if needed)
        selected_subfolder = sorted(candidates)[0]
        object_path = os.path.join(object_path, selected_subfolder)
        subfolders = sorted([f for f in os.listdir(object_path) if os.path.isdir(os.path.join(object_path, f))])
        
        object_path_dir= os.path.dirname(object_path)
        print("show object_path_dir:", object_path_dir)
        test_img= os.listdir(os.path.join(object_path, subfolders[0]))
        print("show test_img:", test_img)
        # object_name = f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.



        feature_object_path = os.path.join(feature_model_path, object_id)
        feature_image_path = os.path.join(feature_vis_path, object_id)
        os.makedirs(feature_object_path, exist_ok=True)
        os.makedirs(feature_image_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        print("show output_path:", output_path)

 
        
   
        # Check if we have the expected 40 subfolders
        #assert len(subfolders) == 40
        images = []
        normals = []
        poses = []
        masks = []
        depths = []
        cameras = []
        object_data = {}
        
        
       
        normals, depths, images, masks, poses, cameras = process_subfolders_parallel(
        subfolders, object_path, object_name, get_intri, imread, 
        max_workers=8, method='process')



        # if len(images) == 0:
        #     # remove dirname of /media/lei/ExtremeSSD/G-objaverse/armor/1090/5456862
        #     continue





        # Get feature model
        imgs = np.stack(images, axis=0)
        dpts = np.stack(depths, axis=0)
        msks = np.stack(masks, axis=0)

        

        try:
            # feature_point_cloud = fuse_feature_rgbd(backbone, imgs, dpts, msks, cameras, None,os.path.join(feature_image_path, f"{object_name}.png"))
            feature_point_cloud = fuse_feature_rgbd_batch(backbone, imgs, dpts, msks, cameras, None,os.path.join(feature_image_path, f"{object_name}.png"))



        except Exception as e:
            print(f"Error processing feature model for {object_name}: {str(e)}")
            # exit(0)
            continue
        ######################################################
        # feature_vis = vis_pca(feature_point_cloud[:, 3:])

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(feature_point_cloud[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(feature_vis)
        # o3d.io.write_point_cloud(os.path.join(feature_object_path,f"{object_name}.pcd"), pcd)
        # print("show feature_object_path:",feature_object_path)
        # o3d.visualization.draw_geometries([pcd])
        ######################################################
        torch.cuda.empty_cache()
        
        # Store bounding box, camera intrinsic and feature 3d points once for the object (use the bbox from the first view)
        object_data['bbox'] = poses[0]['bbox']
        object_data['cam_k'] = poses[0]['cam_k']
        object_data['feature_points'] = feature_point_cloud.astype(np.float32)
        
  
        # Store data for each view
        for k in range(len(images)):
            object_data[f'rgb_{k}'] = images[k]
            object_data[f'mask_{k}'] = masks[k]
            object_data[f'depth_{k}'] = depths[k]
            object_data[f'c2w_{k}'] = poses[k]['c2w']
            object_data[f'fov_{k}'] = poses[k]['fov']
            object_data[f'nrm_{k}'] = normals[k]

        category_data[object_name] = object_data
        
        safe_filename = shape_name_save.replace('/', '_')
        print("show shape_name_save:",safe_filename)# 1000/5003641
        output_file = os.path.join(output_path, f"{safe_filename}.h5")


        
        with h5py.File(output_file, 'w') as h5_file:
            # Create group for this object
            obj_grp = h5_file.create_group(safe_filename)
            # Save all object data
            for key, value in object_data.items():
                obj_grp.create_dataset(key, data=value, compression='gzip', compression_opts=4)



    return output_path


def get_all_folders(root):
    all_folders = []
    categrey = os.listdir(root)
    for item in categrey:
        if not os.path.isdir(f'{root}/{item}'):
            continue
        folders = os.listdir(f'{root}/{item}')
        all_folders += [f'{root}/{item}/{folder}' for folder in folders]
    return all_folders


def merge_h5py_files(category_files, output_path):
    """
    Merge multiple h5py files into a single file.
    """
    with h5py.File(output_path, 'w') as dest_file:
        for category_file in category_files:
            with h5py.File(category_file, 'r') as source_file:
                # Copy all groups and datasets from source to destination
                for name in source_file:
                    source_file.copy(name, dest_file)
    
    print(f"Merged all category files into {output_path}")
    return output_path

import gc  # Add this import
def clear_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        print(f"CUDA Memory cleared: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")


def run():



    # category= 'backpack'

    base_dir =  "/home/xyz/student/lei/shape_canonicalization_gui_tool/data"# "/media/lei/Extreme SSD/Canonicalization/categories20"
    handle_category='toy' # 'bed' #'Christmas_tree' #'doll' # 'Lego' #'car_(automobile)' # 'backpack' # 'airplane' #'person' 


    input_file1= '/home/xyz/student/lei/shape_canonicalization_gui_tool/data_downloader/core_data/folders_1.txt'
    input_file2 ='/home/xyz/student/lei/shape_canonicalization_gui_tool/data_downloader/core_data/2folders.txt'

    with open(input_file1, "r") as f:
        folders1 = [line.strip() for line in f]


    with open(input_file2, "r") as f:
        folders2 = [line.strip() for line in f]


    merged_folders = list(set(folders1 + folders2))


    base_folders = [
        "airplane", "armor", "backpack", "cake", "castle", "chair", "Christmas_tree",
        "dog", "fish", "flower_arrangement", "helmet", "Lego", "person",
        "robot", "space_shuttle", "sword", "toy", "tree","car_(automobile)","statue_(sculpture)"
    ]
   

    # dir = '/home/lei/Documents/Dataset/Canonicalization' # toy


    # for folder in base_folders:
        # category_dir = os.path.join(base_dir, folder)
        # category='airplane' # cellphone_lvis # chair_compare

    id2model = "helper/gobjaverse_280k_index_to_objaverse.json"
    gobjaverse_id = "helper/gobjaverse_id.json"

    with open(id2model, "r") as f:
        id_info = json.load(f)

    with open(gobjaverse_id, "r") as f:
        gob_id = json.load(f)

    dir= base_dir
    data_root = dir + '/G-objaverse'  # Root directory containing category folders
    save_output_dir = dir + '/G-objaverse_h5py_files_v1'  # Directory for output h5py files
    feature_model_dir = dir+ '/G-objaverse_feature_model_v1'
    feature_image_dir = dir + '/G-objaverse_feature_image_v1'
    os.makedirs(save_output_dir, exist_ok=True)
    os.makedirs(feature_image_dir, exist_ok=True)
    os.makedirs(feature_model_dir, exist_ok=True)
    categories = [os.path.basename(f) for f in glob.glob(data_root+'/*')]

    print("see categories:",categories)
    # exit(0)
    

    
    category_files = []
    for category in categories:
        if category  in merged_folders:
            continue
        # if category!=handle_category:
        #     continue


        print("show category:",category)
        # exit(0)
        category_path = os.path.join(data_root, category)
        if os.path.isdir(category_path):
            feature_model_path = os.path.join(feature_model_dir, category)
            feature_image_path = os.path.join(feature_image_dir, category)
            output_dir = os.path.join(save_output_dir, category)           
            result_file = process_category(category_path, output_dir, feature_model_path, feature_image_path,id_info) # dino lifting
            category_files.append(result_file)
        else:
            print(f"Warning: Category directory {category_path} not found.")
    
    # Merge all category files
    # if category_files:
    #     merge_h5py_files(category_files, os.path.join(output_dir, "all_categories.h5"))
    # else:
    #     print("No category files were created. Check your input data.")
