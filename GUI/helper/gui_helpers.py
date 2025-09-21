


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import threading
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import glob
import h5py
import re
import signal
import sys
from pytorch3d.ops import knn_points, knn_gather
import torch
from helper.feat_pc_modules import fuse_feature_rgbd, vis_pca, apply_pca_and_store_colors
import time
import json
import os
from tkinter import filedialog, messagebox






def cluster_and_reorder(features, num_clusters=5):
    """
    Cluster point clouds by their mean features and reorder them based on cluster similarity.

    Args:
        features (torch.Tensor): Shape (B, N, F) feature point clouds.
        num_clusters (int): Number of clusters for K-means.

    Returns:
        reordered_indices (np.ndarray): Indices of reordered clusters.
        tsne_results (np.ndarray): t-SNE projected points (B, 2).
        cluster_labels (np.ndarray): Cluster assignments (B,).
    """
    B, N, F = features.shape

    # Step 1: Compute mean feature for each point cloud
    mean_features = features.mean(axis=1)# Shape (B, F)

    # Step 2: Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=2)
    cluster_labels = kmeans.fit_predict(mean_features)  # Cluster assignments (B,)

    # Step 3: Compute t-SNE for visualization and distance-based reordering
    tsne = TSNE(n_components=2, perplexity=min(2, B-1), random_state=42)
    tsne_results = tsne.fit_transform(mean_features)  # Shape (B, 2)

    # Step 4: Reorder clusters by their mean t-SNE position
    cluster_centroids = np.array([tsne_results[cluster_labels == i].mean(axis=0) for i in range(num_clusters)])
    cluster_order = np.argsort(cluster_centroids[:, 0])  # Sort by X-axis

    # Step 5: Reorder points within each cluster based on distance to cluster centroid
    reordered_indices = []
    reoredered_labels = []
    for cluster in cluster_order:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_mean = cluster_centroids[cluster]
        distances = np.linalg.norm(tsne_results[cluster_indices] - cluster_mean, axis=1)
        sorted_cluster_indices = cluster_indices[np.argsort(distances)]
        reoredered_labels.extend([len(reordered_indices)])
        reordered_indices.extend(sorted_cluster_indices)
    reoredered_labels.extend([len(reordered_indices)])

    return np.array(reordered_indices), tsne_results, cluster_labels, np.array(reoredered_labels)



def create_flexible_grid(spacing, n_shapes):
        """
        Create a flexible grid layout that can accommodate any number of shapes.
        The last row can be partially filled.
        
        Args:
            spacing (float): Spacing between grid positions
            n_shapes (int): Number of shapes to arrange
        
        Returns:
            np.ndarray: Grid centers of shape (n_shapes, 3)
        """
        if n_shapes == 0:
            return np.array([]).reshape(0, 3)
        
        if n_shapes == 1:
            return np.array([[0, 0, 0]], dtype=np.float32)
        
        # Calculate optimal grid dimensions
        # Try to make it as square as possible, but allow rectangular grids
        cols = int(np.ceil(np.sqrt(n_shapes)))
        rows = int(np.ceil(n_shapes / cols))
        
        # Alternative: You could also use a more rectangular layout if preferred
        # rows = int(np.sqrt(n_shapes))
        # cols = int(np.ceil(n_shapes / rows))
        
        print(f"Arranging {n_shapes} shapes in {rows}x{cols} grid")
        
        # Create grid positions
        grid_positions = []
        for row in range(rows):
            for col in range(cols):
                if len(grid_positions) >= n_shapes:
                    break
                x = col * spacing
                y = row * spacing
                z = 0
                grid_positions.append([x, y, z])
            if len(grid_positions) >= n_shapes:
                break
        
        return np.array(grid_positions[:n_shapes], dtype=np.float32)

        
def create_array(x, rows,cols):
    # Create an array with the pattern [0, nx, 0] for n in range(1, rows+1)
    array = np.array([[j*x, n * x, 0] for n in range(1, rows + 1) for j in range(1,cols+1)], dtype=np.float32)
    return array


def furthest_point_sampling(points, num_samples):
    """
    Perform Furthest Point Sampling (FPS) on a set of points.
    
    Parameters:
        points (torch.Tensor): Tensor of shape (N, 3) containing the 3D coordinates.
        num_samples (int): Desired number of keypoints.
    
    Returns:
        sampled_points (torch.Tensor): Tensor of shape (num_samples, 3) of the sampled points.
        sampled_indices (torch.Tensor): Indices of the sampled points in the input tensor.
    """
    N = points.shape[0]
    points = points.cuda()
    device = points.device
    # Initialize array to hold indices of sampled points
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    # Initialize distances to infinity
    distances = torch.full((N,), float('inf'), device=device)
    
    # Randomly select the first point
    sampled_indices[0] = torch.randint(0, N, (1,), device=device)
    
    for i in range(1, num_samples):
        cur_point = points[sampled_indices[i-1]].unsqueeze(0)  # shape (1, 3)
        # Compute Euclidean distances from the current point to all points
        dist = torch.norm(points - cur_point, dim=1)
        # Update the minimum distance to any sampled point so far
        distances = torch.minimum(distances, dist)
        # Select the point with the maximum distance to the current set of sampled points
        sampled_indices[i] = torch.argmax(distances)

    points = points.cpu()
    sampled_indices = sampled_indices.cpu()
    return points[sampled_indices], sampled_indices



def remove_outliers(points, radius=0.15, min_neighbors=10, k=30):
    """
    Remove outliers from a point cloud using PyTorch3D's knn_points.
    
    Parameters:
        points (torch.Tensor): Tensor of shape (N, 3) containing the 3D coordinates.
        radius (float): The radius within which neighbors are counted.
        min_neighbors (int): Minimum number of neighbors (within radius) required for a point to be an inlier.
        k (int): Number of neighbors to consider (should be > min_neighbors).
    
    Returns:
        inlier_points (torch.Tensor): Filtered tensor with outliers removed.
        mask (torch.BoolTensor): Boolean mask for inlier points.
    """
    # Expand points to shape (1, N, 3) for knn_points
    pts = torch.from_numpy(points).unsqueeze(0).cuda()
    # Compute k-nearest neighbors for each point (including itself)
    knn = knn_points(pts, pts, K=k+1)
    # Extract squared distances; shape: (1, N, K+1)
    dists = knn.dists.squeeze(0).cpu().numpy()  # now shape (N, k+1)
    # Remove the first column (distance of each point to itself is 0)
    dists = dists[:, 1:]
    # Count neighbors with squared distance less than radius**2
    neighbor_count = (dists < (radius**2)).sum(axis=-1)
    # Create a mask for points having at least min_neighbors within the radius
    mask = neighbor_count >= min_neighbors
    return points[mask], mask






@dataclass
class DinoShapeData:
    """Data structure for DINO processed shape data"""
    kpts_data: np.ndarray  # Shape: (N, num_points, 387) - N shapes, each with points+features
    kpts_data_view_all: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    tsne_results: Optional[np.ndarray] = None
    cluster_centers: Optional[np.ndarray] = None  # Center shapes for each cluster
    selected_cluster: int = -1
    cluster_center_rotations: Optional[Dict[int, np.ndarray]] = None  
    viz_data: Optional[Dict[str, np.ndarray]] = None
    kpts_data_name_list:Optional[str] = None
    shape_rotations: Optional[Dict[str, np.ndarray]] = None  # TODO This is NEW class variable, please store indentity matrix if they are not rotated



class ShapeProcessor:
    """Backend processing class for shape canonicalization pipeline"""
    
    def __init__(self):
        self.dino_data: Optional[DinoShapeData] = None

        self.result_path=None

        self.category=None
        
    def load_dino_directory(self, h5py_files_path: str, category_path_original_no_need:str) -> DinoShapeData:
        """Load processed DINO features from directory"""
        # This is a placeholder - you'll implement your actual loading logic
        # For now, simulate the expected data structure
        try:
            # Replace this with your actual directory loading code
            # Expected to return kpts_data with shape (N, num_points, 387)
            # where [:,:,:3] are coordinates and [:,:,3:] are DINO features

            # Placeholder implementation - replace with your actual code
            # np.random.seed(42)
            # # Create temporary directory for extraction
            # os.makedirs('temp', exist_ok=True)
                                                             # /media/lei/ExtremeSSD/G-objaverse_h5py_files_v1/armor
            print("input h5py_files_path:", h5py_files_path) # /media/lei/Extreme SSD/Canonicalization/categories20/G-objaverse_h5py_files_v1/armor
            category_path=h5py_files_path
            # category_path= '/home/lei/Documents/Dataset/Canonicalization/G-objaverse/chairr'
            # print("input category_path_original:", category_path_original)

            # input h5py_files_path: /media/lei/Extreme SSD/Canonicalization/categories20/G-objaverse_h5py_files_v1/tree
            # input category_path_original: /media/lei/Extreme SSD/Canonicalization/categories20/G-objaverse/tree
            # make a result folder:
            #  G-objaverse
            parent_dir = os.path.dirname(h5py_files_path)
            parent_dir_res= os.path.basename(parent_dir)+"_results"

          

            self.result_path = os.path.join(os.path.dirname(parent_dir), parent_dir_res)
            # print("show self.result_path:",self.result_path)
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path, exist_ok=True)
            
            category_name = os.path.basename(category_path)
            self.category= category_name
            


            # print("show parent_dir_res:",parent_dir_res)
            # print("show object_folders[0]:",object_folders[0])
            # /media/lei/ExtremeSSD/G-objaverse_h5py_files_v1/armor/1000_5003641.h5
            # exit(0)


            # object_folders = [f for f in os.listdir(sub_folders) if os.path.isdir(os.path.join(category_path, f))]
            category_data = []
            kpts_data = []
            kpts_data_name_list = []
            category_data_pos = []  



  
            def load_feature_point_cloud(h5_file_path, object_name):
                
                with h5py.File(h5_file_path, 'r') as h5_file:

                    # print(list(h5_file.keys()))# ['armor_6524']
                    # exit(0)
                    
                    object_data = h5_file[object_name]
                   
                    if 'feature_points' in object_data:
                        feature_point_cloud = object_data['feature_points'][:]
                        return feature_point_cloud                        
                    else:
                        print(f"Feature point cloud not found for object: {object_name}")
                        return None
                    
                    
                    
            def create_array(x, rows,cols):
                # Create an array with the pattern [0, nx, 0] for n in range(1, rows+1)
                array = np.array([[j*x, n * x, 0] for n in range(1, rows + 1) for j in range(1,cols+1)], dtype=np.float32)
                return array
            
            # num=16
            # Create h5py file for this category

            work_shapes_num= 1000
            shape_name_save_list=[]

            shape_name_save_list_path= "./"+str(category_name)+"_shape_anno_save_list.txt"
            if os.path.exists(shape_name_save_list_path) and os.path.getsize(shape_name_save_list_path) > 0:
                with open(shape_name_save_list_path, 'r') as f:
                    shape_name_save_list = [line.strip() for line in f.readlines()]
            else:
                shape_name_save_list = []

            print("len of done:", len(shape_name_save_list))
            work_shapes_done=len(shape_name_save_list)


            object_folders = sorted(glob.glob(category_path+'/*'))

            # remove files exists in shape_name_save_list



            object_folders_udpated=[]
            for idx, obj_folder in enumerate(tqdm(object_folders, desc=f"filtering  {category_name}")):
                parts = obj_folder.strip("/").split("/")
                # print("show an parts: ", parts) 
                shape_name_done = parts[-1].split('.')[0].replace('_', '/') 

                # print("show an shape_name_done: ", shape_name_done) 
                if shape_name_done in shape_name_save_list:
                    continue

                object_folders_udpated.append(obj_folder)

            print("show  len of object_folders_udpated: ", len(object_folders_udpated))
            print("show  len object_folders: ", len(object_folders))  



                




            for i, obj_folder in enumerate(tqdm(object_folders_udpated, desc=f"Processing {category_name}")):


                # print("show an obj_folder: ", obj_folder)  # SSD/Canonicalization/categories20/G-objaverse/armor/1013/5068563
                parts = obj_folder.strip("/").split("/")
                # print("show an parts: ", parts) 
                # object_name= parts[-1].split('.')[0]
                shape_name_save = parts[-1].split('.')[0].replace('_', '/') # parts[-1].split('.')[0].replace('_', '/') # 

                print("show an shape_name_save: ", shape_name_save)# 1270/6350171

                
                object_name = f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.
                h5_file_path = obj_folder #os.path.join(h5py_files_path, f"{object_name}.h5")


                print("show name:", shape_name_save)# 1000/5003641
                if shape_name_save in shape_name_save_list:

                    continue

  
                
                

           
                if os.path.exists(h5_file_path):
                    print("show h5_file_path:",h5_file_path)
                    try: 
                        feature_point_cloud = load_feature_point_cloud(h5_file_path, parts[-1].split('.')[0]) # entry point cloud 
                    except:
                        print("show errr--------------------------------------------------------------:",h5_file_path)
                        continue
                    print("show feature_point_cloud dim:", feature_point_cloud.shape) #  (4096, 387)     
 
                    # Remove outliers from the feature point cloud
                    inlier_points, mask = remove_outliers(feature_point_cloud[:,:3], radius=0.1, min_neighbors=5, k=50)
                    inlier = feature_point_cloud[mask]
                    category_data.append(inlier)                    
                    # Extract keypoints using FPS
                    _,kpts_data_idx = furthest_point_sampling(torch.from_numpy(inlier[:,:3]).float(), 2048)
                    kpts_data.append(inlier[kpts_data_idx,:])
                    kpts_data_name_list.append(shape_name_save)

                    shape_name_save_list.append(shape_name_save)
                    #category_data_pos.append(feature_point_cloud)

                # if len(category_data)>= num:
                #     print(len(category_data))
                #     break

                if i>= work_shapes_num:
                    break
          


            with open(shape_name_save_list_path, 'w') as f:
                    for item in shape_name_save_list:
                        f.write(f"{item}\n")

            kpts_data = np.stack(kpts_data, axis=0) # 16, 2048, 387) 
            print("show shape of kpts_data:",kpts_data.shape) #  (20, 2048, 387)
          
            dino_data = DinoShapeData(kpts_data=kpts_data)
            dino_data.cluster_center_rotations = {}
            dino_data.shape_rotations= {}
            dino_data.kpts_data_name_list= kpts_data_name_list

            
            dino_data.kpts_data_view_all=kpts_data[:, :, :3].copy()


            for shape_name in kpts_data_name_list:
                dino_data.shape_rotations[shape_name] = np.eye(3)

            
            return dino_data
            
        except Exception as e:
            raise Exception(f"Failed to load DINO data: {str(e)}")
    
    def cluster_and_reorder_tobe_del(self, features: np.ndarray, num_clusters: int, method: str = 'kmeans'):
        """
        Your clustering function - replace with actual implementation
        Expected to return: reordered_indices, tsne_results, cluster_labels, reordered_labels
        """
        # Reshape features for clustering (N_shapes, features) 
        n_shapes, n_points, n_features = features.shape
        features_flat = features.reshape(n_shapes, -1)  # Flatten each shape's features
        
        # Apply clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=num_clusters, random_state=42)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clusterer.fit_predict(features_flat)
        
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_shapes-1))
        tsne_results = tsne.fit_transform(features_flat)
        
        # Reorder by clusters
        reordered_indices = np.argsort(cluster_labels)
        reordered_labels = cluster_labels[reordered_indices]
        
        return reordered_indices, tsne_results, cluster_labels, reordered_labels
    

    def cluster_and_reorder(self, features, num_clusters=5,  method: str = 'kmeans'):
        """
        Cluster point clouds by their mean features and reorder them based on cluster similarity.

        Args:
            features (torch.Tensor): Shape (B, N, F) feature point clouds.
            num_clusters (int): Number of clusters for K-means.

        Returns:
            reordered_indices (np.ndarray): Indices of reordered clusters.
            tsne_results (np.ndarray): t-SNE projected points (B, 2).
            cluster_labels (np.ndarray): Cluster assignments (B,).
        """
        B, N, F = features.shape

        # Step 1: Compute mean feature for each point cloud
        mean_features = features.mean(axis=1)# Shape (B, F)

        # Step 2: Apply K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(mean_features)  # Cluster assignments (B,)

        # Step 3: Compute t-SNE for visualization and distance-based reordering
        tsne = TSNE(n_components=2, perplexity=min(30, B-1), random_state=42)
        tsne_results = tsne.fit_transform(mean_features)  # Shape (B, 2)

        # Step 4: Reorder clusters by their mean t-SNE position
        cluster_centroids = np.array([tsne_results[cluster_labels == i].mean(axis=0) for i in range(num_clusters)])
        cluster_order = np.argsort(cluster_centroids[:, 0])  # Sort by X-axis

        # Step 5: Reorder points within each cluster based on distance to cluster centroid
        reordered_indices = []
        reoredered_labels = []
        for cluster in cluster_order:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_mean = cluster_centroids[cluster]
            distances = np.linalg.norm(tsne_results[cluster_indices] - cluster_mean, axis=1)
            sorted_cluster_indices = cluster_indices[np.argsort(distances)]
            reoredered_labels.extend([len(reordered_indices)])
            reordered_indices.extend(sorted_cluster_indices)
        reoredered_labels.extend([len(reordered_indices)])

        return np.array(reordered_indices), tsne_results, cluster_labels, np.array(reoredered_labels)

    
    def get_cluster_centers(self, kpts_data: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """Get center shapes for each cluster"""
        unique_labels = np.unique(cluster_labels)
        cluster_centers = []
        
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            cluster_mask = cluster_labels == label
            cluster_shapes = kpts_data[cluster_mask]
            
            # Compute centroid shape (mean of all shapes in cluster)
            center_shape = np.mean(cluster_shapes, axis=0)
            cluster_centers.append(center_shape)
        
        return np.array(cluster_centers)
    
    def apply_rotation(self, points: np.ndarray, axis: str, angle_degrees: float) -> np.ndarray:
        """Apply rotation to points around specified axis"""
        angle_rad = np.radians(angle_degrees)
        
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        return np.dot(points, rotation_matrix.T)
