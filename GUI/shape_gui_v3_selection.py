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
import signal
import sys
from pytorch3d.ops import knn_points, knn_gather
import torch
from helper.feat_pc_modules import fuse_feature_rgbd, vis_pca, apply_pca_and_store_colors
import time
import json
import os
from tkinter import filedialog, messagebox


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




@dataclass
class DinoShapeData:
    """Data structure for DINO processed shape data"""
    kpts_data: np.ndarray  # Shape: (N, num_points, 387) - N shapes, each with points+features
    cluster_labels: Optional[np.ndarray] = None
    tsne_results: Optional[np.ndarray] = None
    cluster_centers: Optional[np.ndarray] = None  # Center shapes for each cluster
    selected_cluster: int = -1
    cluster_center_rotations: Optional[Dict[int, np.ndarray]] = None  # NEW
    viz_data: Optional[Dict[str, np.ndarray]] = None
    kpts_data_name_list:Optional[str] = None


class ShapeProcessor:
    """Backend processing class for shape canonicalization pipeline"""
    
    def __init__(self):
        self.dino_data: Optional[DinoShapeData] = None
        
    def load_dino_directory(self, h5py_files_path: str, category_path_original:str) -> DinoShapeData:
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
            print("input h5py_files_path:", h5py_files_path)
            category_path=category_path_original
            # category_path= '/home/lei/Documents/Dataset/Canonicalization/G-objaverse/chairr'


            category_name = os.path.basename(category_path)
            object_folders = sorted(glob.glob(category_path+'/*/*'))
            # object_folders = [f for f in os.listdir(sub_folders) if os.path.isdir(os.path.join(category_path, f))]
            category_data = []
            kpts_data = []
            kpts_data_name_list = []
            category_data_pos = []            
            def load_feature_point_cloud(h5_file_path, object_name):
                with h5py.File(h5_file_path, 'r') as h5_file:
                    # Access the object data using the object name
                    # print("Top-level keys in HDF5 file:")
                    # print(list(h5_file.keys()))
                    object_data = h5_file[object_name]
                    # print("object_data",list(object_data.keys()))
                    # Extract the feature point cloud (assuming it was saved as 'feature_points') # feature_points
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
            for i, obj_folder in enumerate(tqdm(object_folders, desc=f"Processing {category_name}")):
                
                object_name = f"{category_name}_{i}"  # e.g., chair_1, chair_2, etc.
                h5_file_path = os.path.join(h5py_files_path, f"{object_name}.h5")




                # print(len(category_data))
                # print("h5_file_path:",h5_file_path)
                # print("object_name:",object_name)
                # print("show category_path:",category_path) # /home/lei/Documents/Dataset/Canonicalization/G-objaverse/chairr
                # exit(0)
           
                if os.path.exists(h5_file_path):
                    # print("show h5_file_path:",h5_file_path)
                    feature_point_cloud = load_feature_point_cloud(h5_file_path, object_name) # entry point cloud 
                    print("show feature_point_cloud dim:",feature_point_cloud.shape) #  (4096, 387)             
                    # Remove outliers from the feature point cloud
                    inlier_points, mask = remove_outliers(feature_point_cloud[:,:3], radius=0.1, min_neighbors=5, k=50)
                    inlier = feature_point_cloud[mask]
                    category_data.append(inlier)                    
                    # Extract keypoints using FPS
                    _,kpts_data_idx = furthest_point_sampling(torch.from_numpy(inlier[:,:3]).float(), 2048)
                    kpts_data.append(inlier[kpts_data_idx,:])
                    kpts_data_name_list.append(object_name)
                    #category_data_pos.append(feature_point_cloud)

                # if len(category_data)>= num:
                #     print(len(category_data))
                #     break


            kpts_data = np.stack(kpts_data, axis=0) # 16, 2048, 387) 
            print("show shape of kpts_data:",kpts_data.shape) #  (20, 2048, 387)
            # exit()
            dino_data = DinoShapeData(kpts_data=kpts_data)
            dino_data.cluster_center_rotations = {}
            dino_data.kpts_data_name_list= kpts_data_name_list
            
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

class ShapeCanonicalizeGUI:
    """Main GUI application for 3D shape canonicalization pipeline"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Shape Canonicalization Pipeline")
        self.root.geometry("1600x1000")
        
        self.processor = ShapeProcessor()
        self.vis = None
        
        # New attributes for DINO workflow
        self.rotation_angle = 15.0  # degrees per rotation step

        self.cluster_tabs = {}  # {cluster_id: {'tab': tab_widget, 'fig': figure, 'ax': axis, 'canvas': canvas}}
        self.active_cluster_tabs = set()  # Track which clusters have open tabs
    
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel for visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_control_panel(left_panel)
        self.setup_visualization_panel(right_panel)
    
    def setup_control_panel(self, parent):
        """Setup the control panel with buttons and options"""
        # File operations
        file_frame = ttk.LabelFrame(parent, text="Data Loading", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))


        # Category path original selection
        ttk.Label(file_frame, text="Original Category Path:").pack(anchor=tk.W)
        self.category_path_original = tk.StringVar(value="No path selected")
        
        category_path_frame = ttk.Frame(file_frame)
        category_path_frame.pack(fill=tk.X, pady=2)

        ttk.Entry(category_path_frame, textvariable=self.category_path_original, 
             state="readonly", width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(category_path_frame, text="Browse", 
                command=self.select_category_path_original).pack(side=tk.RIGHT)


        # DINO directory selection
        ttk.Label(file_frame, text="DINO Features Directory:").pack(anchor=tk.W, pady=(10, 0))
        self.dino_directory_path = tk.StringVar(value="No path selected")
        
        dino_path_frame = ttk.Frame(file_frame)
        dino_path_frame.pack(fill=tk.X, pady=2)
        
        ttk.Entry(dino_path_frame, textvariable=self.dino_directory_path, 
                state="readonly", width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(dino_path_frame, text="Browse", 
                command=self.select_dino_directory).pack(side=tk.RIGHT)
        




        # Cluster directory selection
        ttk.Label(file_frame, text="Save Cluster Canonicalization Directory:").pack(anchor=tk.W, pady=(10, 0))
        self.res_save_directory_path = tk.StringVar(value="No path selected")
        
        res_save_directory_path_frame = ttk.Frame(file_frame)
        res_save_directory_path_frame.pack(fill=tk.X, pady=2)
        
        ttk.Entry(res_save_directory_path_frame, textvariable=self.res_save_directory_path, 
                state="readonly", width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(res_save_directory_path_frame, text="Browse", 
                command=self.select_res_save_directory).pack(side=tk.RIGHT)

        
        # ttk.Button(file_frame, text="Load DINO Directory",  command=self.load_dino_directory).pack(fill=tk.X, pady=2)
        # path_str = str(self.dino_directory_path.get())


        ttk.Button(
            file_frame,
            text="Load DINO Directory",
            command=lambda: self.load_dino_directory(str(self.category_path_original.get()), str(self.dino_directory_path.get()))
        ).pack(fill=tk.X, pady=2)
                

        ttk.Button(file_frame, text="Save Current Result",  command=self.save_result).pack(fill=tk.X, pady=2)
        
        # DINO Clustering operations
        dino_frame = ttk.LabelFrame(parent, text="DINO Shape Clustering", padding=10)
        dino_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Number of clusters
        ttk.Label(dino_frame, text="Number of Clusters:").pack(anchor=tk.W)
        self.num_clusters = tk.IntVar(value=4)
        cluster_spin = ttk.Spinbox(dino_frame, from_=2, to=20, width=10, 
                                  textvariable=self.num_clusters)
        cluster_spin.pack(fill=tk.X, pady=2)
        
        # Clustering method for DINO
        ttk.Label(dino_frame, text="Clustering Method:").pack(anchor=tk.W)
        self.dino_cluster_method = tk.StringVar(value="kmeans")
        method_frame = ttk.Frame(dino_frame)
        method_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(method_frame, text="K-Means", variable=self.dino_cluster_method, 
                       value="kmeans").pack(side=tk.LEFT)
        ttk.Radiobutton(method_frame, text="DBSCAN", variable=self.dino_cluster_method, 
                       value="dbscan").pack(side=tk.LEFT)
        
        ttk.Button(dino_frame, text="Run Re-clustering", command=self.run_dino_clustering).pack(fill=tk.X, pady=5)
        
        # Cluster Selection and Visualization
        cluster_viz_frame = ttk.LabelFrame(parent, text="Cluster Visualization", padding=10)
        cluster_viz_frame.pack(fill=tk.X, pady=(0, 10))
        
        # ttk.Button(cluster_viz_frame, text="Show Cluster Centers", 
        #           command=self.show_cluster_centers).pack(fill=tk.X, pady=2)
        
        # Cluster selection
        ttk.Label(cluster_viz_frame, text="Select Cluster:").pack(anchor=tk.W)
        self.selected_cluster = tk.IntVar(value=0)

        self.cluster_selector = ttk.Combobox(cluster_viz_frame, textvariable=self.selected_cluster, state="readonly", width=15)
        self.cluster_selector.pack(fill=tk.X, pady=2)
        self.cluster_selector.bind('<<ComboboxSelected>>', self.on_cluster_selected)
        
        # ttk.Button(cluster_viz_frame, text="Show Selected Cluster", 
        #           command=self.show_selected_cluster).pack(fill=tk.X, pady=2)

        print("show self.selected_cluster:",self.selected_cluster)
        # exit(0)
        
        # Shape Rotation Controls
        # rotation_frame = ttk.LabelFrame(parent, text="Shape Rotation", padding=10)
        # rotation_frame.pack(fill=tk.X, pady=(0, 10))
        # ttk.Label(rotation_frame, text="Rotation Angle (degrees):").pack(anchor=tk.W)
        # self.rotation_angle_var = tk.DoubleVar(value=90.0)
        # ttk.Scale(rotation_frame, from_=5, to=90, variable=self.rotation_angle_var,orient=tk.HORIZONTAL, ).pack(fill=tk.X, pady=2) # resolution=5
        # # Rotation buttons
        # rot_buttons_frame = ttk.Frame(rotation_frame)
        # rot_buttons_frame.pack(fill=tk.X, pady=5)
        # ttk.Button(rot_buttons_frame, text="Rotate X", 
        #           command=lambda: self.rotate_selected_cluster('x')).pack(side=tk.LEFT, padx=2)
        # ttk.Button(rot_buttons_frame, text="Rotate Y", 
        #           command=lambda: self.rotate_selected_cluster('y')).pack(side=tk.LEFT, padx=2)
        # ttk.Button(rot_buttons_frame, text="Rotate Z", 
        #           command=lambda: self.rotate_selected_cluster('z')).pack(side=tk.LEFT, padx=2)
        # ttk.Button(rotation_frame, text="Reset Rotation", 
        #           command=self.reset_cluster_rotation).pack(fill=tk.X, pady=2)
        
        # Visualization options
        vis_frame = ttk.LabelFrame(parent, text="Visualization", padding=10)
        vis_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(vis_frame, text="Show All Shapes", command=self.show_all_shapes).pack(fill=tk.X, pady=2)
        ttk.Button(vis_frame, text="Show t-SNE Plot",   command=self.show_tsne_plot).pack(fill=tk.X, pady=2)
        ttk.Button(vis_frame, text="Reset View",  command=self.reset_3d_view).pack(fill=tk.X, pady=2)
        
        # Status
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_text = tk.Text(status_frame, height=6, width=30)
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Advanced operations
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Operations", padding=10)
        advanced_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(advanced_frame, text="Further Processing", 
                  command=self.further_processing).pack(fill=tk.X, pady=2)
        ttk.Button(advanced_frame, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=2)
    

    def select_category_path_original(self):
        """Select the original category path directory"""
        directory_path = filedialog.askdirectory(title="Select Original Category Directory")
        if directory_path:
            self.category_path_original.set(directory_path)
            self.log_status(f"Original category path selected: {os.path.basename(directory_path)}")

    def select_dino_directory(self):
        """Select the DINO features directory"""
        directory_path = filedialog.askdirectory(title="Select DINO Features Directory")
        if directory_path:
            self.dino_directory_path.set(directory_path)
            self.log_status(f"DINO features path selected: {os.path.basename(directory_path)}")


    def select_res_save_directory(self):
        """Select Result Saving Directory"""
        directory_path = filedialog.askdirectory(title="Select Result Saving Directory")
        if directory_path:
            self.res_save_directory_path.set(directory_path)
            self.log_status(f"Result saving path selected: {os.path.basename(directory_path)}")



# select_res_save_directory

    def setup_visualization_panel(self, parent):
        """Setup the visualization panel with embedded 3D view"""
        # Create notebook for tabbed visualization
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 3D Visualization tab
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="3D Visualization")
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Setup 3D visualization
        self.setup_3d_visualization()
        
        # Setup statistics panel
        self.setup_statistics_panel()
    
    def setup_3d_visualization(self):
        """Setup the embedded 3D visualization"""
        # Create matplotlib 3D figure
        self.fig_3d = plt.figure(figsize=(12, 10))
        # self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d', 
                                        position=[0, 0, 1, 1])  # [left, bottom, width, height]
        

        self.fig_3d.subplots_adjust(left=0, right=1, top=1, bottom=0, 
                               wspace=0, hspace=0)
        
        # Embed in tkinter
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, self.viz_frame)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True,   padx=0, pady=0)  # Remove padding
        
        # Add toolbar for 3D navigation
        toolbar_frame = ttk.Frame(self.viz_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.toolbar_3d = NavigationToolbar2Tk(self.canvas_3d, toolbar_frame)
        self.toolbar_3d.update()
        
        # Initial setup
        # self.ax_3d.set_xlabel('X')
        # self.ax_3d.set_ylabel('Y')
        # self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Point Cloud Visualization')
        self.ax_3d.set_axis_off()
        
        # Show instructions initially
        self.show_instructions()
    
    def setup_statistics_panel(self):
        """Setup the statistics visualization panel"""
        # Create matplotlib figure for statistics
        self.fig_stats = plt.figure(figsize=(10, 8))
        self.ax_stats = self.fig_stats.add_subplot(111)
        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, self.stats_frame)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial instructions
        self.ax_stats.text(0.5, 0.5, 'Run clustering to see statistics', 
                          ha='center', va='center', transform=self.ax_stats.transAxes, 
                          fontsize=14)
        self.ax_stats.axis('off')
        self.canvas_stats.draw()
    
    def show_instructions(self):
        """Show initial instructions in the 3D view"""
        self.ax_3d.clear()
        self.ax_3d.text(0.5, 0.5, 0.5, 
                       '3D Shape Canonicalization Pipeline\n\n' +
                       'Instructions:\n' +
                       '1. Load DINO processed directory\n' +
                       '2. Set number of clusters\n' +
                       '3. Run re-clustering\n' +
                       '4. View cluster centers\n' +
                       '5. Select and rotate clusters',
                       transform=self.ax_3d.transAxes,
                       fontsize=12, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        

        self.ax_3d.set_axis_off()
        self.ax_3d.grid(False)
        # self.add_coordinate_system()
        self.ax_3d.set_title('3D Point Cloud Visualization')
        # self.fig_3d.tight_layout(pad=0)
        self.canvas_3d.draw()


    def add_coordinate_system(self):
        """Add a small coordinate system indicator in the bottom-right corner"""
        # Position closer to center (moved from bottom-right corner)
        origin_x, origin_y, origin_z = 0.75, 0.2, 0.2  # Moved closer to center
        arrow_length = 0.15  # Increased length from 0.08 to 0.15
        
        # Define the three axes
        axes_data = [
            # X-axis (red)
            ([origin_x, origin_x + arrow_length], [origin_y, origin_y], [origin_z, origin_z], 'red', 'X'),
            # Y-axis (green) 
            ([origin_x, origin_x], [origin_y, origin_y + arrow_length], [origin_z, origin_z], 'green', 'Y'),
            # Z-axis (blue)
            ([origin_x, origin_x], [origin_y, origin_y], [origin_z, origin_z + arrow_length], 'blue', 'Z')
        ]
        
        # Draw each axis
        for x_coords, y_coords, z_coords, color, label in axes_data:
            # Draw the line
            self.ax_3d.plot(x_coords, y_coords, z_coords, 
                        color=color, linewidth=4, alpha=0.9,  # Increased linewidth and alpha
                        transform=self.ax_3d.transAxes)
            
            # Add label at the end
            end_x, end_y, end_z = x_coords[1], y_coords[1], z_coords[1]
            self.ax_3d.text(end_x + 0.02, end_y + 0.02, end_z + 0.02, label,  # Offset labels slightly
                        color=color, fontsize=12, fontweight='bold',  # Larger font
                        transform=self.ax_3d.transAxes)
        
    def log_status(self, message: str):
        """Log a status message"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def load_dino_directory(self, category_path:str,dino_path:str ):
        """Load DINO processed data from directory"""
        # directory_path = filedialog.askdirectory(title="Select DINO Data Directory")


        # category_path = self.category_path_original.get()
        # dino_path = self.dino_directory_path.get()

        print("show category_path:",category_path)
        print("show dino_path:",dino_path)

        
        
        if dino_path:
            try:
                self.log_status(f"Loading DINO data from: {os.path.basename(dino_path)}")
                self.processor.dino_data = self.processor.load_dino_directory(dino_path, category_path)
                
                n_shapes = len(self.processor.dino_data.kpts_data)
                n_points = self.processor.dino_data.kpts_data.shape[1]
                self.log_status(f"Loaded {n_shapes} shapes with {n_points} points each")
                print(f"Loaded {n_shapes} shapes with {n_points} points each")
                # Show all shapes initially
                self.show_all_shapes()
                self.add_coordinate_system()
                self.notebook.select(0)  # Switch to 3D visualization tab
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load DINO data: {str(e)}")
                self.log_status(f"Error loading directory: {str(e)}")
    
    def run_dino_clustering(self):
        """Run clustering on DINO features"""
        if self.processor.dino_data is None:
            messagebox.showwarning("Warning", "Please load DINO data first")
            return
        
        num_clusters = self.num_clusters.get()
        method = self.dino_cluster_method.get()
        
        self.log_status(f"Running {method} clustering with {num_clusters} clusters...")
        
        try:
            # Extract DINO features (everything except first 3 coordinates)

            kpts_data= self.processor.dino_data.kpts_data
            features = kpts_data[:, :, 3:]
            num= features.shape[0]

            self.show_re_clustering_tsne_view(num_clusters)
        except Exception as e:
            messagebox.showerror("Error", f"Clustering failed: {str(e)}")
            self.log_status(f"Clustering error: {str(e)}")





    

    
    # def show_all_shapes(self):
    #     """Show all shapes in the dataset"""
    #     if self.processor.dino_data is None:
    #         return
        

    #     self.ax_3d.clear()

    #     kpts_data = self.processor.dino_data.kpts_data
    #     print("show shape ------------------ kpts_data:",kpts_data.shape)
    #     # kpts_data: (16, 2048, 387)
    #     n_shapes = len(kpts_data)

    #     num_clusters = 1
    #     reordered_indices, tsne_results, cluster_labels,reordered_labels = cluster_and_reorder(kpts_data[:,:,3:], num_clusters=num_clusters)
    #     tsne_center = np.zeros_like(kpts_data[:n_shapes,0,:3])
    #     tsne_center[:,:2] = tsne_results[:n_shapes]


    #     all_colors = apply_pca_and_store_colors(kpts_data,True)[:n_shapes].reshape(-1,3)
    #     all_points = kpts_data[:n_shapes, :, :3]

    #     # n_shapes= 20

    #     col = int(np.sqrt(n_shapes))
    #     K = all_points.shape[1]
    #     center = create_array(2,col,col)[:,None,:].repeat(K, axis=1) # grid centers
    #     all_points_ = kpts_data[:n_shapes,:,:3].reshape(-1,3) + center.reshape(-1,3)
    #     all_points_ = all_points_.reshape(-1,3)
    #     all_colors_ = all_colors





    #     self.ax_3d.scatter(all_points_[:, 0], all_points_[:, 1], all_points_[:, 2],
    #                  c=all_colors_, s=1, alpha=0.6)
        
    #     self.ax_3d.grid(False)
    #     self.ax_3d.set_axis_off()

    #     # self.ax_3d.set_xlabel('X')
    #     # self.ax_3d.set_ylabel('Y')
    #     # self.ax_3d.set_zlabel('Z')
    #     self.ax_3d.set_title('All Loaded Shapes(Gird View)')
    #     self.canvas_3d.mpl_connect('scroll_event', self.on_scroll)
        
    #     # Set equal aspect ratio
    #     x_range = all_points_[:, 0].max() - all_points_[:, 0].min()
    #     y_range = all_points_[:, 1].max() - all_points_[:, 1].min()
    #     z_range = all_points_[:, 2].max() - all_points_[:, 2].min()
    #     max_range = max(x_range, y_range, z_range)
        
    #     x_center = (all_points_[:, 0].max() + all_points_[:, 0].min()) / 2
    #     y_center = (all_points_[:, 1].max() + all_points_[:, 1].min()) / 2
    #     z_center = (all_points_[:, 2].max() + all_points_[:, 2].min()) / 2
        
    #     self.ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
    #     self.ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
    #     self.ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
    #     self.fig_3d.tight_layout(pad=0)
    #     self.canvas_3d.draw()


    def show_all_shapes(self):
        """Show all shapes in the dataset - FIXED for any number of shapes"""
        if self.processor.dino_data is None:
            return
        
        self.ax_3d.clear()

        kpts_data = self.processor.dino_data.kpts_data
        print("show shape ------------------ kpts_data:", kpts_data.shape)
        n_shapes = len(kpts_data)

        num_clusters = 1
        reordered_indices, tsne_results, cluster_labels, reordered_labels = cluster_and_reorder(
            kpts_data[:, :, 3:], num_clusters=num_clusters
        )
        tsne_center = np.zeros_like(kpts_data[:n_shapes, 0, :3])
        tsne_center[:, :2] = tsne_results[:n_shapes]

        all_colors = apply_pca_and_store_colors(kpts_data, True)[:n_shapes].reshape(-1, 3)
        all_points = kpts_data[:n_shapes, :, :3]

        # FIXED: Flexible grid layout for any number of shapes
        spacing = 2.0  # Adjust this value to change spacing between shapes
        grid_centers = create_flexible_grid(spacing, n_shapes)
        
        print(f"Grid centers shape: {grid_centers.shape}")
        print(f"First few grid positions: {grid_centers[:5]}")
        
        K = all_points.shape[1]  # Number of points per shape
        
        # Expand grid centers to match the number of points per shape
        grid_centers_expanded = grid_centers[:, None, :].repeat(K, axis=1)  # Shape: (n_shapes, K, 3)
        
        # Add grid positions to all points
        all_points_positioned = all_points + grid_centers_expanded
        all_points_ = all_points_positioned.reshape(-1, 3)
        all_colors_ = all_colors

        # Plot all shapes
        self.ax_3d.scatter(all_points_[:, 0], all_points_[:, 1], all_points_[:, 2],
                        c=all_colors_, s=1, alpha=0.6)
        
        self.ax_3d.grid(False)
        self.ax_3d.set_axis_off()
        self.ax_3d.set_title(f'All Loaded Shapes - Grid View ({n_shapes} shapes)')
        self.canvas_3d.mpl_connect('scroll_event', self.on_scroll)
        
        # Set equal aspect ratio
        x_range = all_points_[:, 0].max() - all_points_[:, 0].min()
        y_range = all_points_[:, 1].max() - all_points_[:, 1].min()
        z_range = all_points_[:, 2].max() - all_points_[:, 2].min()
        max_range = max(x_range, y_range, z_range)
        
        x_center = (all_points_[:, 0].max() + all_points_[:, 0].min()) / 2
        y_center = (all_points_[:, 1].max() + all_points_[:, 1].min()) / 2
        z_center = (all_points_[:, 2].max() + all_points_[:, 2].min()) / 2
        
        self.ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
        self.ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
        self.ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        self.fig_3d.tight_layout(pad=0)
        self.canvas_3d.draw()

    # # ALTERNATIVE: If you want to keep using your original create_array function, here's a fixed version:
    # def create_array_flexible(spacing, n_shapes):
    #     """
    #     Fixed version of create_array that handles any number of shapes
    #     """
    #     if n_shapes == 0:
    #         return np.array([]).reshape(0, 3)
        
    #     # Calculate grid dimensions to fit all shapes
    #     cols = int(np.ceil(np.sqrt(n_shapes)))
    #     rows = int(np.ceil(n_shapes / cols))
        
    #     print(f"Creating {rows}x{cols} grid for {n_shapes} shapes")
        
    #     # Generate positions for all needed shapes
    #     positions = []
    #     for row in range(rows):
    #         for col in range(cols):
    #             if len(positions) >= n_shapes:
    #                 break
    #             x = (col + 1) * spacing  # +1 to match your original indexing
    #             y = (row + 1) * spacing  # +1 to match your original indexing
    #             z = 0
    #             positions.append([x, y, z])
    #         if len(positions) >= n_shapes:
    #             break
        
    #     return np.array(positions[:n_shapes], dtype=np.float32)

    def show_re_clustering_tsne_view(self,num_clusters ):
        """Show all shapes in the dataset"""
        if self.processor.dino_data is None:
            return
        try:
            kpts_data = self.processor.dino_data.kpts_data
            n_shapes = len(kpts_data)
            
            self.log_status(f"Re-clustering {n_shapes} shapes into {num_clusters} clusters...")
            
            # Perform clustering and reordering
            reordered_indices, tsne_results, cluster_labels, reordered_labels = cluster_and_reorder(
                kpts_data[:, :, 3:], num_clusters=num_clusters
            )
            
            # Scale t-SNE results for better visualization
            tsne_scale = 0.15 # Adjustable parameter
            tsne_results_scaled = tsne_results * tsne_scale
            
            # Create t-SNE center positions for shapes
            tsne_center = np.zeros_like(kpts_data[:n_shapes, 0, :3])
            tsne_center[:, :2] = tsne_results_scaled[:n_shapes]
            
            # Store clustering results in dino_data
            self.processor.dino_data.cluster_labels = cluster_labels
            self.processor.dino_data.tsne_results = tsne_results_scaled
            
            # Prepare visualization data
            self.processor.dino_data.viz_data = {
                'all_colors': apply_pca_and_store_colors(kpts_data, True)[:n_shapes].reshape(-1, 3),
                'all_points_positioned': kpts_data[:n_shapes, :, :3].reshape(-1, 3) + 
                                    tsne_center[:, None, :].repeat(kpts_data.shape[1], axis=1).reshape(-1, 3),
                'tsne_center': tsne_center,
                'n_shapes': n_shapes,
                'points_per_shape': kpts_data.shape[1]
            }
            
            # Get cluster centers with positions
            cluster_centers_data = self.get_cluster_centers_with_positions(kpts_data, cluster_labels, tsne_center)
            self.processor.dino_data.cluster_centers = cluster_centers_data
            
            # Update cluster selector UI
            if cluster_centers_data:
                cluster_ids = sorted(list(cluster_centers_data.keys()))
                self.cluster_selector['values'] = [str(cid) for cid in cluster_ids]
                if cluster_ids:
                    self.cluster_selector.set(str(cluster_ids[0]))
                    self.selected_cluster.set(cluster_ids[0])
                    self.log_status(f"Available clusters: {cluster_ids}")
            
            # Initialize rotation matrices for all clusters
            if not hasattr(self.processor.dino_data, 'cluster_center_rotations'):
                self.processor.dino_data.cluster_center_rotations = {}
            for cluster_id in cluster_centers_data.keys():
                if cluster_id not in self.processor.dino_data.cluster_center_rotations:
                    self.processor.dino_data.cluster_center_rotations[cluster_id] = np.eye(3)
            
            self.log_status(f"Clustering complete: {len(cluster_centers_data)} clusters created")
            # Step 2: Update visualization

            self.viewer_update('tsne')
    
            
        except Exception as e:
            self.log_status(f"Clustering failed: {str(e)}")
      
      

    # def viewer_update(self, view_mode='tsne'):
    #     """Update the 3D visualization with current data
        
    #     Args:
    #         view_mode (str): 'tsne' for t-SNE positioned view, 'grid' for grid layout, 'cluster_only' for cluster centers only
    #     """
    #     # if self.processor.dino_data is None:
    #     #     self.show_instructions()
    #     #     return
        
    #     # Check if we have the required data for visualization
    #     if not hasattr(self.processor.dino_data, 'viz_data') or self.processor.dino_data.viz_data is None:
    #         self.log_status("No visualization data available. Please run clustering first.")
    #         return
        
    #     # Clear the current plot
    #     self.ax_3d.clear()
        
    #     # Get visualization data
    #     viz_data = self.processor.dino_data.viz_data
    #     cluster_centers_data = self.processor.dino_data.cluster_centers
    #     cluster_labels = self.processor.dino_data.cluster_labels
        
    #     # Setup basic plot properties
    #     self.canvas_3d.mpl_connect('scroll_event', self.on_scroll)
    #     self.ax_3d.set_axis_off()
    #     self.ax_3d.grid(False)
        
    #     if view_mode == 'tsne':
    #         # Show all shapes positioned by t-SNE (original shapes, no rotations applied to background)
    #         all_points_ = viz_data['all_points_positioned']
    #         all_colors_ = viz_data['all_colors']
            
    #         # Plot all shapes as background
    #         self.ax_3d.scatter(all_points_[:, 0], all_points_[:, 1], all_points_[:, 2],
    #                         c=all_colors_, s=1, alpha=0.6)
            
    #         # Draw cluster centers with rotations applied
    #         if cluster_centers_data:
    #             self.draw_cluster_centers_and_boxes_with_rotation(cluster_centers_data, cluster_labels)
            
    #         title = 'All Loaded Shapes (t-SNE View) - Centers Show Rotations'
            
    #     elif view_mode == 'grid':
    #         # Show shapes in grid layout (original functionality)
    #         kpts_data = self.processor.dino_data.kpts_data
    #         n_shapes = viz_data['n_shapes']
    #         K = viz_data['points_per_shape']
            
    #         # Create grid layout
    #         col = int(np.sqrt(n_shapes))
    #         grid_centers = create_array(2, col, col)[:n_shapes, None, :].repeat(K, axis=1)
    #         all_points_grid = kpts_data[:n_shapes, :, :3].reshape(-1, 3) + grid_centers.reshape(-1, 3)
    #         all_colors_grid = viz_data['all_colors']
            
    #         self.ax_3d.scatter(all_points_grid[:, 0], all_points_grid[:, 1], all_points_grid[:, 2],
    #                         c=all_colors_grid, s=1, alpha=0.6)
            
    #         title = 'All Loaded Shapes (Grid View)'
            
    #     elif view_mode == 'cluster_only':
    #         # Show only cluster centers with rotations
    #         if cluster_centers_data:
    #             self.draw_cluster_centers_and_boxes_with_rotation(cluster_centers_data) # draw_cluster_centers_and_boxes_with_rotation
    #             title = 'Cluster Centers Only - Showing Rotations'
    #         else:
    #             self.ax_3d.text(0.5, 0.5, 0.5, 'No cluster centers available', 
    #                         transform=self.ax_3d.transAxes, ha='center', va='center')
    #             title = 'No Clusters Available'
        
    #     else:
    #         self.log_status(f"Unknown view mode: {view_mode}")
    #         return
        
    #     # Set title
    #     self.ax_3d.set_title(title)
        
    #     # Calculate and set equal aspect ratio
    #     self.set_equal_aspect_ratio_from_data(view_mode)
        
    #     # Final drawing
    #     self.fig_3d.tight_layout(pad=0)
    #     self.canvas_3d.draw_idle()

    def viewer_update(self, view_mode='tsne'):
        """Update the 3D visualization with current data"""
        if self.processor.dino_data is None:
            self.show_instructions()
            return
        
        if not hasattr(self.processor.dino_data, 'viz_data') or self.processor.dino_data.viz_data is None:
            self.log_status("No visualization data available. Please run clustering first.")
            return
        
        self.ax_3d.clear()
        
        viz_data = self.processor.dino_data.viz_data
        cluster_centers_data = self.processor.dino_data.cluster_centers
        cluster_labels = self.processor.dino_data.cluster_labels
        
        self.canvas_3d.mpl_connect('scroll_event', self.on_scroll)
        self.ax_3d.set_axis_off()
        self.ax_3d.grid(False)
        
        if view_mode == 'tsne':
            # CRITICAL FIX: Exclude cluster center shapes from background
            all_points_, all_colors_ = self.get_background_shapes_excluding_centers(viz_data, cluster_centers_data, cluster_labels)
            
            # Plot background shapes (excluding cluster centers)
            if len(all_points_) > 0:
                self.ax_3d.scatter(all_points_[:, 0], all_points_[:, 1], all_points_[:, 2],
                                c=all_colors_, s=1, alpha=0.6)
            
            # Draw cluster centers with rotations applied (these are the ONLY centers now)
            if cluster_centers_data:
                self.draw_cluster_centers_and_boxes_with_rotation(cluster_centers_data, cluster_labels)
            
            title = 'All Loaded Shapes (t-SNE View) - Centers Show Rotations'
            
        elif view_mode == 'grid':
            # Grid view unchanged
            kpts_data = self.processor.dino_data.kpts_data
            n_shapes = viz_data['n_shapes']
            K = viz_data['points_per_shape']
            
            col = int(np.sqrt(n_shapes))
            grid_centers = create_array(2, col, col)[:n_shapes, None, :].repeat(K, axis=1)
            all_points_grid = kpts_data[:n_shapes, :, :3].reshape(-1, 3) + grid_centers.reshape(-1, 3)
            all_colors_grid = viz_data['all_colors']
            
            self.ax_3d.scatter(all_points_grid[:, 0], all_points_grid[:, 1], all_points_grid[:, 2],
                            c=all_colors_grid, s=1, alpha=0.6)
            
            title = 'All Loaded Shapes (Grid View)'
            
        elif view_mode == 'cluster_only':
            # Show only cluster centers with rotations
            if cluster_centers_data:
                self.draw_cluster_centers_only_with_rotation(cluster_centers_data)
                title = 'Cluster Centers Only - Showing Rotations'
            else:
                self.ax_3d.text(0.5, 0.5, 0.5, 'No cluster centers available', 
                            transform=self.ax_3d.transAxes, ha='center', va='center')
                title = 'No Clusters Available'
        
        else:
            self.log_status(f"Unknown view mode: {view_mode}")
            return
        
        self.ax_3d.set_title(title)
        self.set_equal_aspect_ratio_from_data(view_mode)
        
        self.fig_3d.tight_layout(pad=0)
        self.canvas_3d.draw_idle()



    def get_background_shapes_excluding_centers(self, viz_data, cluster_centers_data, cluster_labels):
        """Get background shapes excluding cluster center shapes to avoid duplication"""
        all_points_positioned = viz_data['all_points_positioned']
        all_colors = viz_data['all_colors']
        points_per_shape = viz_data['points_per_shape']
        
        if not cluster_centers_data:
            return all_points_positioned, all_colors
        
        # Get indices of cluster center shapes (representative shapes)
        center_shape_indices = set()
        for center_data in cluster_centers_data.values():
            representative_idx = center_data.get('\n representative_index', -1)
            if representative_idx >= 0:
                center_shape_indices.add(representative_idx)
        
        if not center_shape_indices:
            return all_points_positioned, all_colors
        
        # Create mask to exclude center shapes
        total_shapes = len(cluster_labels)
        keep_mask = np.ones(total_shapes * points_per_shape, dtype=bool)
        
        for center_idx in center_shape_indices:
            # Exclude all points belonging to this center shape
            start_idx = center_idx * points_per_shape
            end_idx = (center_idx + 1) * points_per_shape
            keep_mask[start_idx:end_idx] = False
        
        # Filter out center shapes from background
        background_points = all_points_positioned[keep_mask]
        background_colors = all_colors[keep_mask]
        
        return background_points, background_colors
    

    # def draw_cluster_centers_and_boxes_with_rotation(self, cluster_centers_data, cluster_labels):
    #     """Draw cluster centers and bounding boxes with rotations applied - REPLACES original centers"""
    #     unique_labels = list(cluster_centers_data.keys())
    #     colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
    #     try:
    #         selected_cluster = int(self.selected_cluster.get())
    #     except:
    #         selected_cluster = -1
        
    #     for i, (label, center_data) in enumerate(cluster_centers_data.items()):
    #         center_shape = center_data['center_shape']
    #         tsne_position = center_data['tsne_position']
            
    #         # CRITICAL: Always apply rotation (even if identity) to get the current state
    #         # This ensures we're drawing the CURRENT center state, not the original
    #         rotated_center_shape = self.apply_rotation_to_cluster_center(center_shape, label)
    #         center_points = rotated_center_shape[:, :3]  # Shape: (2048, 3)
            
    #         # Apply t-SNE positioning
    #         positioned_center_points = center_points + tsne_position
            
    #         # Subsample for better performance
    #         # if len(center_points) > 500:
    #         #     indices = np.random.choice(len(center_points), 500, replace=False)
    #         #     display_points = positioned_center_points[indices]
    #         # else:
    #             # display_points = positioned_center_points
    #         display_points = positioned_center_points
    #         # Plot cluster center with distinct color and size
    #         color = colors[i]
            
    #         # Check if this cluster has been rotated
    #         is_rotated = (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
    #                     label in self.processor.dino_data.cluster_center_rotations and
    #                     not np.allclose(self.processor.dino_data.cluster_center_rotations[label], np.eye(3)))
            
    #         # Highlight selected cluster and rotation status
    #         if label == selected_cluster:
    #             # Selected cluster: larger, brighter, with border
    #             edge_color = 'red' if is_rotated else 'black'
    #             point_size = 10 if is_rotated else 8
    #             alpha = 1.0
    #             label_suffix = ' (ROTATED)*' if is_rotated else '*'
    #         else:
    #             # Other clusters
    #             edge_color = 'darkred' if is_rotated else None
    #             point_size = 7 if is_rotated else 5
    #             alpha = 0.9 if is_rotated else 0.8
    #             label_suffix = ' (ROTATED)' if is_rotated else ''
            
    #         self.ax_3d.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
    #                         c=[color], s=point_size, alpha=alpha, 
    #                         edgecolors=edge_color, linewidths=1.0 if edge_color else 0,
    #                         label=f'Cluster {label}{label_suffix} ({center_data["num_shapes"]} shapes)')
            
    #         # Draw bounding box around cluster center (rotated position)
    #         self.draw_cluster_bounding_box_with_rotation_status(positioned_center_points, color, label)


    def draw_cluster_centers_and_boxes_with_rotation(self, cluster_centers_data, cluster_labels):
        """Draw cluster centers and bounding boxes with rotations applied - using original shape colors"""
        unique_labels = list(cluster_centers_data.keys())
        bbox_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))  # For bounding boxes
        
        try:
            selected_cluster = int(self.selected_cluster.get())
        except:
            selected_cluster = -1
        
        for i, (label, center_data) in enumerate(cluster_centers_data.items()):
            center_shape = center_data['center_shape']
            tsne_position = center_data['tsne_position']
            
            # CRITICAL: Always apply rotation (even if identity) to get the current state
            rotated_center_shape = self.apply_rotation_to_cluster_center(center_shape, label)
            center_points = rotated_center_shape[:, :3]  # Shape: (2048, 3)
            
            # Apply t-SNE positioning
            positioned_center_points = center_points + tsne_position
            display_points = positioned_center_points
            
            # Get ORIGINAL shape colors using PCA on the rotated shape features
            try:
                # Use the full rotated shape (coordinates + features) for PCA coloring
                shape_colors = apply_pca_and_store_colors(
                    rotated_center_shape.reshape(1, -1, rotated_center_shape.shape[-1]), 
                    True
                )[0]  # Shape: (2048, 3)
            except Exception as e:
                print(f"Warning: Could not generate PCA colors for cluster {label}: {e}")
                # Fallback to uniform color based on cluster
                bbox_color = bbox_colors[i]
                shape_colors = np.tile(bbox_color, (len(display_points), 1))
            
            # Check if this cluster has been rotated
            is_rotated = (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
                        label in self.processor.dino_data.cluster_center_rotations and
                        not np.allclose(self.processor.dino_data.cluster_center_rotations[label], np.eye(3)))
            
            # Determine point size and edge colors based on selection and rotation status
            if label == selected_cluster:
                # Selected cluster: larger points, with border
                edge_color = 'red' if is_rotated else 'black'
                point_size = 3 if is_rotated else 2
                alpha = 1.0
                edge_width = 0.5
                label_suffix = ' (ROTATED)*' if is_rotated else '*'
            else:
                # Other clusters: normal size, optional edge
                edge_color = 'darkred' if is_rotated else None
                point_size = 2 if is_rotated else 1
                alpha = 0.9 if is_rotated else 0.8
                edge_width = 0.3 if is_rotated else 0
                label_suffix = ' (ROTATED)' if is_rotated else ''
            
            # Plot cluster center with ORIGINAL shape colors
            self.ax_3d.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                            c=shape_colors, s=point_size, alpha=alpha, 
                            edgecolors=edge_color, linewidths=edge_width,
                            label=f'Cluster {label}{label_suffix} ({center_data["num_shapes"]} shapes)')
            
            # Draw bounding box around cluster center (using bbox color for distinction)
            bbox_color = bbox_colors[i]
            self.draw_cluster_bounding_box_with_rotation_status(positioned_center_points, bbox_color, label)


    def draw_cluster_bounding_box_with_rotation_status(self, points, color, label):
        """Draw a bounding box around cluster center points with rotation status"""
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Check if this cluster has been rotated
        is_rotated = (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
                    label in self.processor.dino_data.cluster_center_rotations and
                    not np.allclose(self.processor.dino_data.cluster_center_rotations[label], np.eye(3)))
        
        # Define the 8 vertices of the bounding box
        vertices = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # 0
            [max_coords[0], min_coords[1], min_coords[2]],  # 1
            [max_coords[0], max_coords[1], min_coords[2]],  # 2
            [min_coords[0], max_coords[1], min_coords[2]],  # 3
            [min_coords[0], min_coords[1], max_coords[2]],  # 4
            [max_coords[0], min_coords[1], max_coords[2]],  # 5
            [max_coords[0], max_coords[1], max_coords[2]],  # 6
            [min_coords[0], max_coords[1], max_coords[2]]   # 7
        ])
        
        # Define the 12 edges of the bounding box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Draw each edge with different style for rotated vs non-rotated
        linewidth = 3 if is_rotated else 2
        linestyle = '-' if is_rotated else '--'
        alpha = 1.0 if is_rotated else 0.7
        
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            self.ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                        color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        
        # Add cluster label at the center of bounding box with rotation indicator
        box_center = (min_coords + max_coords) / 2
        label_text = f'C{label}{"*" if is_rotated else ""}'
        
        self.ax_3d.text(box_center[0], box_center[1], box_center[2] + (max_coords[2] - min_coords[2]) * 0.6,
                    label_text, color=color, fontsize=12, fontweight='bold', 
                    ha='center', va='center')

    def draw_cluster_centers_and_boxes(self, cluster_centers_data, cluster_labels):
        """Draw cluster centers and bounding boxes"""
        unique_labels = list(cluster_centers_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, (label, center_data) in enumerate(cluster_centers_data.items()):
            center_shape = center_data['center_shape']
            tsne_position = center_data['tsne_position']
            
            # Get 3D coordinates of center shape
            center_points = center_shape[:, :3]  # Shape: (2048, 3)
            
            # Apply t-SNE positioning
            positioned_center_points = center_points + tsne_position
            
            # Subsample for better performance
            if len(center_points) > 500:
                indices = np.random.choice(len(center_points), 500, replace=False)
                display_points = positioned_center_points[indices]
            else:
                display_points = positioned_center_points
            
            # Plot cluster center with distinct color and larger size
            color = colors[i]
            self.ax_3d.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                            c=[color], s=5, alpha=0.9, 
                            label=f'Cluster {label} Center ({center_data["num_shapes"]} shapes)')
            
            # Draw bounding box around cluster center
            self.draw_cluster_bounding_box(positioned_center_points, color, label)

    def draw_cluster_bounding_box(self, points, color, label):
        """Draw a bounding box around cluster center points"""
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Define the 8 vertices of the bounding box
        vertices = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # 0
            [max_coords[0], min_coords[1], min_coords[2]],  # 1
            [max_coords[0], max_coords[1], min_coords[2]],  # 2
            [min_coords[0], max_coords[1], min_coords[2]],  # 3
            [min_coords[0], min_coords[1], max_coords[2]],  # 4
            [max_coords[0], min_coords[1], max_coords[2]],  # 5
            [max_coords[0], max_coords[1], max_coords[2]],  # 6
            [min_coords[0], max_coords[1], max_coords[2]]   # 7
        ])
        
        # Define the 12 edges of the bounding box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Draw each edge
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            self.ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                        color=color, linewidth=2, alpha=0.8)
        
        # Add cluster label at the center of bounding box
        box_center = (min_coords + max_coords) / 2
        self.ax_3d.text(box_center[0], box_center[1], box_center[2] + (max_coords[2] - min_coords[2]) * 0.6,
                    f'C{label}', color=color, fontsize=12, fontweight='bold', 
                    ha='center', va='center')



    def draw_cluster_centers_only(self, cluster_centers_data):
        """Draw only cluster center points without background shapes"""
        unique_labels = list(cluster_centers_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        try:
            selected_cluster = int(self.selected_cluster.get())
        except:
            selected_cluster = -1
        
        for i, (label, center_data) in enumerate(cluster_centers_data.items()):
            center_shape = center_data['center_shape']
            tsne_position = center_data['tsne_position']
            
            # Apply rotation directly to center shape
            rotated_center_shape = self.apply_rotation_to_cluster_center(center_shape, label)
            center_points = rotated_center_shape[:, :3]
            
            # Apply t-SNE positioning
            positioned_center_points = center_points + tsne_position
            
            # Subsample for better performance
            if len(center_points) > 500:
                indices = np.random.choice(len(center_points), 500, replace=False)
                display_points = positioned_center_points[indices]
            else:
                display_points = positioned_center_points
            
            color = colors[i]
            
            # Highlight selected cluster
            if label == selected_cluster:
                self.ax_3d.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                                c=[color], s=8, alpha=1.0, 
                                edgecolors='black', linewidths=0.5,
                                label=f'Cluster {label}* ({center_data["num_shapes"]} shapes)')
            else:
                self.ax_3d.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                                c=[color], s=5, alpha=0.7,
                                label=f'Cluster {label} ({center_data["num_shapes"]} shapes)')
        
        # Add legend
        self.ax_3d.legend(loc='upper left', bbox_to_anchor=(0, 1))


    def set_equal_aspect_ratio_from_data(self, view_mode):
        """Set equal aspect ratio based on current view mode data"""
        if view_mode == 'tsne' and hasattr(self.processor.dino_data, 'viz_data'):
            viz_data = self.processor.dino_data.viz_data
            all_points_ = viz_data['all_points_positioned']
            self.set_equal_aspect_ratio_for_points(all_points_)
            
        elif view_mode == 'grid':
            kpts_data = self.processor.dino_data.kpts_data
            viz_data = self.processor.dino_data.viz_data
            n_shapes = viz_data['n_shapes']
            K = viz_data['points_per_shape']
            
            col = int(np.sqrt(n_shapes))
            grid_centers = create_array(2, col, col)[:n_shapes, None, :].repeat(K, axis=1)
            all_points_grid = kpts_data[:n_shapes, :, :3].reshape(-1, 3) + grid_centers.reshape(-1, 3)
            self.set_equal_aspect_ratio_for_points(all_points_grid)
            
        elif view_mode == 'cluster_only' and self.processor.dino_data.cluster_centers:
            # Get bounds from cluster centers with rotations applied
            all_cluster_points = []
            for label, center_data in self.processor.dino_data.cluster_centers.items():
                rotated_center_shape = self.apply_rotation_to_cluster_center(center_data['center_shape'], label)
                center_points = rotated_center_shape[:, :3] + center_data['tsne_position']
                all_cluster_points.append(center_points)
            
            if all_cluster_points:
                combined_points = np.vstack(all_cluster_points)
                self.set_equal_aspect_ratio_for_points(combined_points)



    def set_equal_aspect_ratio_for_points(self, points):
        """Helper method to set equal aspect ratio for given points"""
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        max_range = max(x_range, y_range, z_range)
        
        x_center = (points[:, 0].max() + points[:, 0].min()) / 2
        y_center = (points[:, 1].max() + points[:, 1].min()) / 2
        z_center = (points[:, 2].max() + points[:, 2].min()) / 2
        
        self.ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
        self.ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
        self.ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)



    def get_cluster_centers_with_positions(self, kpts_data, cluster_labels, tsne_center):
        """Get cluster center shapes with their t-SNE positions"""
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels != -1]  # Exclude noise (-1)
        
        cluster_centers_data = {}
        
        for label in valid_labels:
            # Find all shapes in this cluster
            cluster_mask = cluster_labels == label
            cluster_shape_indices = np.where(cluster_mask)[0]
            
            if len(cluster_shape_indices) == 0:
                continue
                
            # Get shapes in this cluster
            cluster_shapes = kpts_data[cluster_mask]  # Shape: (n_shapes_in_cluster, N, 387)
            
            # Method 1: Find the shape closest to the cluster centroid in feature space
            # Extract DINO features and compute mean features per shape (same as in cluster_and_reorder)
            cluster_features = cluster_shapes[:, :, 3:]  # Shape: (n_shapes_in_cluster, N, F)
            cluster_mean_features = cluster_features.mean(axis=1)  # Shape: (n_shapes_in_cluster, F)
            
            # Compute the centroid of mean features for this cluster
            cluster_feature_centroid = cluster_mean_features.mean(axis=0)  # Shape: (F,)
            
            # Find the shape with mean features closest to the cluster centroid
            distances_to_centroid = []
            for i, shape_mean_features in enumerate(cluster_mean_features):
                distance = np.linalg.norm(shape_mean_features - cluster_feature_centroid)
                distances_to_centroid.append(distance)
            
            # Pick the most representative shape (closest to cluster centroid)
            representative_idx_in_cluster = np.argmin(distances_to_centroid)
            center_shape = cluster_shapes[representative_idx_in_cluster]
            
            # Get the actual index in the original dataset
            actual_shape_index = cluster_shape_indices[representative_idx_in_cluster]
            center_tsne_position = tsne_center[actual_shape_index]
            
            # Alternative Method 2: Pick the shape closest to cluster centroid in t-SNE space
            # cluster_tsne_positions = tsne_center[cluster_mask]  # t-SNE positions of shapes in cluster
            # cluster_tsne_centroid = cluster_tsne_positions.mean(axis=0)  # Mean t-SNE position
            # tsne_distances = [np.linalg.norm(pos - cluster_tsne_centroid) for pos in cluster_tsne_positions]
            # representative_idx_in_cluster = np.argmin(tsne_distances)
            # center_shape = cluster_shapes[representative_idx_in_cluster]
            # actual_shape_index = cluster_shape_indices[representative_idx_in_cluster]
            # center_tsne_position = tsne_center[actual_shape_index]
            
            # Store cluster center data
            cluster_centers_data[label] = {
                'center_shape': center_shape,
                'tsne_position': center_tsne_position,
                'shape_indices': cluster_shape_indices,
                'representative_index': actual_shape_index,
                'num_shapes': len(cluster_shape_indices),
                'feature_centroid': cluster_feature_centroid  # Store for potential future use
            }
        
        return cluster_centers_data

    def on_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes != self.ax_3d:
            return
        
        # Get current axis limits
        xlim = self.ax_3d.get_xlim()
        ylim = self.ax_3d.get_ylim()
        zlim = self.ax_3d.get_zlim()
        
        # Calculate zoom factor
        zoom_factor = 1.1 if event.step < 0 else 0.9
        
        # Calculate new limits
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * zoom_factor / 2
        y_range = (ylim[1] - ylim[0]) * zoom_factor / 2
        z_range = (zlim[1] - zlim[0]) * zoom_factor / 2
        
        # Set new limits
        self.ax_3d.set_xlim(x_center - x_range, x_center + x_range)
        self.ax_3d.set_ylim(y_center - y_range, y_center + y_range)
        self.ax_3d.set_zlim(z_center - z_range, z_center + z_range)
        
        self.canvas_3d.draw_idle()
        
    def show_cluster_centers(self):
        """Show cluster center shapes with bounding boxes"""
        if (self.processor.dino_data is None or 
            self.processor.dino_data.cluster_centers is None):
            messagebox.showwarning("Warning", "Please run clustering first")
            return
        
        self.ax_3d.clear()

        self.ax_3d.set_axis_off()
        self.ax_3d.grid(False)

        cluster_centers = self.processor.dino_data.cluster_centers
        cluster_labels = self.processor.dino_data.cluster_labels
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels != -1]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_labels)))
        
        for i, (label, center_shape) in enumerate(zip(valid_labels, cluster_centers)):
            points = center_shape[:, :3]  # Extract coordinates
            
            # Apply any accumulated rotations
            if (self.processor.dino_data.cluster_rotations and 
                label in self.processor.dino_data.cluster_rotations):
                rotation_matrix = self.processor.dino_data.cluster_rotations[label]
                points = np.dot(points, rotation_matrix.T)
            
            # Subsample if needed
            # if len(points) > 1024:
            #     indices = np.random.choice(len(points), 1024, replace=False)
            #     points = points[indices]
            
            # Offset for visualization
            offset_x = (i % 3) * 4
            offset_y = (i // 3) * 4
            points_offset = points + np.array([offset_x, offset_y, 0])
            
            # Plot center shape
            self.ax_3d.scatter(points_offset[:, 0], points_offset[:, 1], points_offset[:, 2],
                              c=[colors[i]], s=2, alpha=0.7, label=f'Cluster {label}')
            
            # Add bounding box
            min_coords = np.min(points_offset, axis=0)
            max_coords = np.max(points_offset, axis=0)
            
            # Draw bounding box edges
            self.draw_bounding_box(min_coords, max_coords, colors[i])
        
        # self.ax_3d.set_xlabel('X')
        # self.ax_3d.set_ylabel('Y')
        # self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_axis_off()
        self.ax_3d.set_title('Cluster Centers with Bounding Boxes')
        self.ax_3d.legend()
        
        self.canvas_3d.draw_idle() 
    
    def draw_bounding_box(self, min_coords, max_coords, color):
        """Draw a bounding box around a cluster center"""
        # Define the 8 vertices of the bounding box
        vertices = [
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ]
        
        # Define the 12 edges of the bounding box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Draw each edge
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            self.ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                           color=color, linewidth=2, alpha=0.8)
    
    def show_selected_cluster__(self):
        """Show the selected cluster center shape"""
        if (self.processor.dino_data is None or 
            self.processor.dino_data.cluster_centers is None):
            messagebox.showwarning("Warning", "Please run clustering first")
            return
        
        selected_cluster = self.selected_cluster.get()
        cluster_labels = self.processor.dino_data.cluster_labels
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels != -1]
        
        if selected_cluster not in valid_labels:
            messagebox.showwarning("Warning", f"Cluster {selected_cluster} not found")
            return
        
        self.ax_3d.clear()
        
        # Find the cluster center
        cluster_idx = np.where(valid_labels == selected_cluster)[0][0]
        center_shape = self.processor.dino_data.cluster_centers[cluster_idx]
        points = center_shape[:, :3]
        
        # Apply any accumulated rotations
        if (self.processor.dino_data.cluster_rotations and 
            selected_cluster in self.processor.dino_data.cluster_rotations):
            rotation_matrix = self.processor.dino_data.cluster_rotations[selected_cluster]
            points = np.dot(points, rotation_matrix.T)
        
        # Plot the shape
        self.ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c='blue', s=2, alpha=0.7)
        
        # self.ax_3d.set_xlabel('X')
        # self.ax_3d.set_ylabel('Y')
        # self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_axis_off()
        self.ax_3d.set_title(f'Cluster {selected_cluster} Center Shape')
        
        # Set equal aspect ratio
        self.set_equal_aspect_3d(points)
        
        self.canvas_3d.draw()
        
        # Update selected cluster in data
        self.processor.dino_data.selected_cluster = selected_cluster
    
    def show_selected_cluster_with_rotation(self):
        """Show the selected cluster center shape with current rotation applied"""
        if (self.processor.dino_data is None or 
            not hasattr(self.processor.dino_data, 'cluster_centers') or
            self.processor.dino_data.cluster_centers is None):
            messagebox.showwarning("Warning", "Please run clustering first")
            return
        try:
            selected_cluster = int(self.selected_cluster.get())  # Ensure it's an integer
        except:
            messagebox.showwarning("Warning", "Invalid cluster selection")
            return
        
        if selected_cluster not in self.processor.dino_data.cluster_centers:
            messagebox.showwarning("Warning", f"Cluster {selected_cluster} not found")
            return
        
        # # Add coordinate system in corner
        self.add_coordinate_system()

        


    def set_equal_aspect_3d(self, points):
        """Set equal aspect ratio for 3D plot"""
        # Get the range of each axis
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        
        # Find the maximum range
        max_range = max(x_range, y_range, z_range)
        
        # Set the limits to create equal aspect ratio
        x_center = (points[:, 0].max() + points[:, 0].min()) / 2
        y_center = (points[:, 1].max() + points[:, 1].min()) / 2
        z_center = (points[:, 2].max() + points[:, 2].min()) / 2
        
        self.ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
        self.ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
        self.ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)
    


    def on_cluster_selected(self, event):
        """Handle cluster selection change - create/switch to cluster tab"""
        print(f"Cluster selected: {self.selected_cluster.get()}")  # Debug print
        
        try:
            selected_cluster = int(self.selected_cluster.get())
        except:
            messagebox.showwarning("Warning", "Invalid cluster selection")
            return
        
        # Create or switch to cluster tab
        self.create_or_switch_cluster_tab(selected_cluster)



    def create_or_switch_cluster_tab(self, cluster_id):
        """Create a new tab for the cluster or switch to existing one"""
        if (self.processor.dino_data is None or 
            not hasattr(self.processor.dino_data, 'cluster_centers') or
            self.processor.dino_data.cluster_centers is None):
            messagebox.showwarning("Warning", "Please run clustering first")
            return
        
        if cluster_id not in self.processor.dino_data.cluster_centers:
            messagebox.showwarning("Warning", f"Cluster {cluster_id} not found")
            return
        
        # Check if tab already exists
        if cluster_id in self.cluster_tabs:
            # Switch to existing tab
            tab_index = self.notebook.index(self.cluster_tabs[cluster_id]['tab'])
            self.notebook.select(tab_index)
            self.log_status(f"Switched to existing Cluster {cluster_id} tab")
        else:
            # Create new tab
            self.create_cluster_tab(cluster_id)
            self.log_status(f"Created new tab for Cluster {cluster_id}")


    def create_cluster_tab(self, cluster_id):
        """Create a new tab dedicated to visualizing a specific cluster"""
        # Create new tab frame
        cluster_tab = ttk.Frame(self.notebook)
        tab_name = f"Cluster {cluster_id}"
        self.notebook.add(cluster_tab, text=tab_name)
        
        # Create main container with controls and visualization
        main_container = ttk.Frame(cluster_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for cluster-specific controls
        control_panel = ttk.Frame(main_container, width=250)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_panel.pack_propagate(False)
        
        # Right panel for visualization
        viz_panel = ttk.Frame(main_container)
        viz_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup cluster-specific controls
        self.setup_cluster_controls(control_panel, cluster_id)
        
        # Setup cluster visualization
        fig, ax, canvas, toolbar = self.setup_cluster_visualization(viz_panel, cluster_id)
        
        # Store tab information
        self.cluster_tabs[cluster_id] = {
            'tab': cluster_tab,
            'fig': fig,
            'ax': ax,
            'canvas': canvas,
            'toolbar': toolbar,
            'control_panel': control_panel,
            'viz_panel': viz_panel
        }
        
        self.active_cluster_tabs.add(cluster_id)
        
        # Initial visualization
        self.update_cluster_tab_visualization(cluster_id)
        
        # Switch to the new tab
        tab_index = self.notebook.index(cluster_tab)
        self.notebook.select(tab_index)


 

    # def setup_cluster_controls(self, parent, cluster_id):
    #     """
    #     Updated setup controls specific to a cluster tab - includes canonicalization functionality
    #     """
    #     # Cluster info
    #     info_frame = ttk.LabelFrame(parent, text=f"Cluster {cluster_id} Info", padding=10)
    #     info_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     # Get cluster data
    #     center_data = self.processor.dino_data.cluster_centers[cluster_id]
        
    #     info_text = f"Shapes in cluster: \n {center_data['num_shapes']}\n"
    #     # info_text += f"Representative index: {center_data.get('representative_index', 'N/A')}"
        
    #     info_label = ttk.Label(info_frame, text=info_text, font=('TkDefaultFont', 9))
    #     info_label.pack(anchor=tk.W)
        
    #     # View controls
    #     view_frame = ttk.LabelFrame(parent, text="View Options", padding=10)
    #     view_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     # View mode for this cluster
    #     cluster_view_var = tk.StringVar(value="center_only")
    #     setattr(self, f'cluster_{cluster_id}_view_var', cluster_view_var)
        
    #     ttk.Radiobutton(view_frame, text="Center Only", variable=cluster_view_var, value="center_only", 
    #                 command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
    #     ttk.Radiobutton(view_frame, text="All Shapes in Cluster", variable=cluster_view_var, value="all_shapes", 
    #                 command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
        
    #     # Rotation controls specific to this cluster
    #     rotation_frame = ttk.LabelFrame(parent, text="Center Rotation Controls", padding=10)
    #     rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     # Rotation angle for this cluster
    #     ttk.Label(rotation_frame, text="Rotation Angle:").pack(anchor=tk.W)
    #     cluster_rotation_var = tk.DoubleVar(value=90.0)
    #     ttk.Scale(rotation_frame, from_=5, to=90, variable=cluster_rotation_var,
    #             orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
    #     # Store the rotation variable for this cluster
    #     setattr(self, f'cluster_{cluster_id}_rotation_var', cluster_rotation_var)
        
    #     # Rotation buttons
    #     rot_buttons_frame = ttk.Frame(rotation_frame)
    #     rot_buttons_frame.pack(fill=tk.X, pady=5)
        
    #     ttk.Button(rot_buttons_frame, text="Rotate X", 
    #             command=lambda: self.rotate_cluster_in_tab(cluster_id, 'x')).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(rot_buttons_frame, text="Rotate Y", 
    #             command=lambda: self.rotate_cluster_in_tab(cluster_id, 'y')).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(rot_buttons_frame, text="Rotate Z", 
    #             command=lambda: self.rotate_cluster_in_tab(cluster_id, 'z')).pack(side=tk.LEFT, padx=2)
        
    #     ttk.Button(rotation_frame, text="Reset Center Rotation", 
    #             command=lambda: self.reset_cluster_rotation_in_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
    #     # NEW: Canonicalization controls
    #     canonicalization_frame = ttk.LabelFrame(parent, text="Cluster Canonicalization", padding=10)
    #     canonicalization_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     ttk.Label(canonicalization_frame, text="Align all shapes to cluster center:", 
    #             font=('TkDefaultFont', 9)).pack(anchor=tk.W, pady=(0, 5))
        
    #     ttk.Button(canonicalization_frame, text="Canonicalize Cluster", 
    #             command=lambda: self.canonicalize_cluster_shapes(cluster_id)).pack(fill=tk.X, pady=2)
    #     ttk.Button(canonicalization_frame, text="Reset Shape Alignments", 
    #             command=lambda: self.reset_cluster_canonicalization(cluster_id)).pack(fill=tk.X, pady=2)
        
    #     # Progress indicator for canonicalization
    #     canonicalization_progress_var = tk.StringVar(value="Ready")
    #     setattr(self, f'cluster_{cluster_id}_canonicalization_status', canonicalization_progress_var)
        
    #     progress_label = ttk.Label(canonicalization_frame, textvariable=canonicalization_progress_var, 
    #                             font=('TkDefaultFont', 8), foreground='blue')
    #     progress_label.pack(anchor=tk.W, pady=(5, 0))
        
    #     # Actions
    #     action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
    #     action_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     ttk.Button(action_frame, text="Export Cluster", 
    #             command=lambda: self.export_cluster(cluster_id)).pack(fill=tk.X, pady=2)
    #     ttk.Button(action_frame, text="Close Tab", 
    #             command=lambda: self.close_cluster_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
    #     # Status for this cluster
    #     status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
    #     status_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     cluster_status_text = tk.Text(status_frame, height=6, width=25)
    #     cluster_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=cluster_status_text.yview)
    #     cluster_status_text.configure(yscrollcommand=cluster_scrollbar.set)
    #     cluster_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    #     cluster_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    #     # Store status text widget for this cluster
    #     setattr(self, f'cluster_{cluster_id}_status_text', cluster_status_text)


    def setup_cluster_controls(self, parent, cluster_id):
        """Setup controls specific to a cluster tab with mouse selection"""
        return self.setup_cluster_controls_with_selection(parent, cluster_id)


    # def setup_cluster_controls_with_selection(self, parent, cluster_id):
    #     """
    #     Updated setup controls with mouse selection functionality
    #     """
    #     # Cluster info
    #     info_frame = ttk.LabelFrame(parent, text=f"Cluster {cluster_id} Info", padding=10)
    #     info_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     # Get cluster data
    #     center_data = self.processor.dino_data.cluster_centers[cluster_id]
        
    #     info_text = f"Shapes in cluster: {center_data['num_shapes']}\n"
    #     info_text += f"Representative index: {center_data.get('representative_index', 'N/A')}"
        
    #     info_label = ttk.Label(info_frame, text=info_text, font=('TkDefaultFont', 9))
    #     info_label.pack(anchor=tk.W)
        
    #     # NEW: Shape Selection Info
    #     selection_info_frame = ttk.LabelFrame(parent, text="Shape Selection", padding=10)
    #     selection_info_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     # Selected shape display
    #     selected_shape_var = tk.StringVar(value="No shape selected")
    #     setattr(self, f'cluster_{cluster_id}_selected_shape', selected_shape_var)
        
    #     ttk.Label(selection_info_frame, text="Selected Shape:").pack(anchor=tk.W)
    #     selected_shape_label = ttk.Label(selection_info_frame, textvariable=selected_shape_var, 
    #                                 font=('TkDefaultFont', 9, 'bold'), foreground='blue')
    #     selected_shape_label.pack(anchor=tk.W, pady=(0, 5))
        
    #     # Selection controls
    #     selection_controls_frame = ttk.Frame(selection_info_frame)
    #     selection_controls_frame.pack(fill=tk.X, pady=5)
        
    #     ttk.Button(selection_controls_frame, text="Clear Selection", 
    #             command=lambda: self.clear_shape_selection(cluster_id)).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(selection_controls_frame, text="Highlight Selected", 
    #             command=lambda: self.highlight_selected_shape(cluster_id)).pack(side=tk.LEFT, padx=2)
        
    #     # Enable/disable selection mode
    #     selection_mode_var = tk.BooleanVar(value=True)
    #     setattr(self, f'cluster_{cluster_id}_selection_mode', selection_mode_var)
        
    #     ttk.Checkbutton(selection_info_frame, text="Enable shape selection (click to select)", 
    #                 variable=selection_mode_var,
    #                 command=lambda: self.toggle_selection_mode(cluster_id)).pack(anchor=tk.W, pady=5)
        
    #     # View controls
    #     view_frame = ttk.LabelFrame(parent, text="View Options", padding=10)
    #     view_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     # View mode for this cluster
    #     cluster_view_var = tk.StringVar(value="center_only")
    #     setattr(self, f'cluster_{cluster_id}_view_var', cluster_view_var)
        
    #     ttk.Radiobutton(view_frame, text="Center Only", variable=cluster_view_var, value="center_only", 
    #                 command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
    #     ttk.Radiobutton(view_frame, text="All Shapes in Cluster", variable=cluster_view_var, value="all_shapes", 
    #                 command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
        
    #     # Rotation controls specific to this cluster
    #     rotation_frame = ttk.LabelFrame(parent, text="Center Rotation Controls", padding=10)
    #     rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     # Rotation angle for this cluster
    #     ttk.Label(rotation_frame, text="Rotation Angle:").pack(anchor=tk.W)
    #     cluster_rotation_var = tk.DoubleVar(value=90.0)
    #     ttk.Scale(rotation_frame, from_=5, to=90, variable=cluster_rotation_var,
    #             orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
    #     # Store the rotation variable for this cluster
    #     setattr(self, f'cluster_{cluster_id}_rotation_var', cluster_rotation_var)
        
    #     # Rotation buttons
    #     rot_buttons_frame = ttk.Frame(rotation_frame)
    #     rot_buttons_frame.pack(fill=tk.X, pady=5)
        
    #     ttk.Button(rot_buttons_frame, text="Rotate X", 
    #             command=lambda: self.rotate_cluster_in_tab(cluster_id, 'x')).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(rot_buttons_frame, text="Rotate Y", 
    #             command=lambda: self.rotate_cluster_in_tab(cluster_id, 'y')).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(rot_buttons_frame, text="Rotate Z", 
    #             command=lambda: self.rotate_cluster_in_tab(cluster_id, 'z')).pack(side=tk.LEFT, padx=2)
        
    #     ttk.Button(rotation_frame, text="Reset Center Rotation", 
    #             command=lambda: self.reset_cluster_rotation_in_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
    #     # NEW: Selected Shape Rotation Controls
    #     selected_rotation_frame = ttk.LabelFrame(parent, text="Selected Shape Controls", padding=10)
    #     selected_rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     ttk.Label(selected_rotation_frame, text="Rotate Selected Shape:").pack(anchor=tk.W)
    #     selected_rot_buttons_frame = ttk.Frame(selected_rotation_frame)
    #     selected_rot_buttons_frame.pack(fill=tk.X, pady=5)
        
    #     ttk.Button(selected_rot_buttons_frame, text="Rotate X", 
    #             command=lambda: self.rotate_selected_shape(cluster_id, 'x')).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(selected_rot_buttons_frame, text="Rotate Y", 
    #             command=lambda: self.rotate_selected_shape(cluster_id, 'y')).pack(side=tk.LEFT, padx=2)
    #     ttk.Button(selected_rot_buttons_frame, text="Rotate Z", 
    #             command=lambda: self.rotate_selected_shape(cluster_id, 'z')).pack(side=tk.LEFT, padx=2)
        
    #     ttk.Button(selected_rotation_frame, text="Reset Selected Shape", 
    #             command=lambda: self.reset_selected_shape_rotation(cluster_id)).pack(fill=tk.X, pady=2)
        
    #     # Canonicalization controls
    #     canonicalization_frame = ttk.LabelFrame(parent, text="Cluster Canonicalization", padding=10)
    #     canonicalization_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     ttk.Label(canonicalization_frame, text="Align all shapes to cluster center:", 
    #             font=('TkDefaultFont', 9)).pack(anchor=tk.W, pady=(0, 5))
        
    #     ttk.Button(canonicalization_frame, text="Canonicalize Cluster", 
    #             command=lambda: self.canonicalize_cluster_shapes(cluster_id)).pack(fill=tk.X, pady=2)
    #     ttk.Button(canonicalization_frame, text="Reset Shape Alignments", 
    #             command=lambda: self.reset_cluster_canonicalization(cluster_id)).pack(fill=tk.X, pady=2)
        
    #     # Progress indicator for canonicalization
    #     canonicalization_progress_var = tk.StringVar(value="Ready")
    #     setattr(self, f'cluster_{cluster_id}_canonicalization_status', canonicalization_progress_var)
        
    #     progress_label = ttk.Label(canonicalization_frame, textvariable=canonicalization_progress_var, 
    #                             font=('TkDefaultFont', 8), foreground='blue')
    #     progress_label.pack(anchor=tk.W, pady=(5, 0))
        
    #     # Actions
    #     action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
    #     action_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     ttk.Button(action_frame, text="Export Cluster", 
    #             command=lambda: self.export_cluster(cluster_id)).pack(fill=tk.X, pady=2)
    #     ttk.Button(action_frame, text="Close Tab", 
    #             command=lambda: self.close_cluster_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
    #     # Status for this cluster
    #     status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
    #     status_frame.pack(fill=tk.X, pady=(0, 10))
        
    #     cluster_status_text = tk.Text(status_frame, height=6, width=25)
    #     cluster_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=cluster_status_text.yview)
    #     cluster_status_text.configure(yscrollcommand=cluster_scrollbar.set)
    #     cluster_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    #     cluster_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    #     # Store status text widget for this cluster
    #     setattr(self, f'cluster_{cluster_id}_status_text', cluster_status_text)

    def setup_cluster_controls_with_selection(self, parent, cluster_id):
        """
        Updated setup controls with mouse selection functionality
        """
        # Cluster info
        info_frame = ttk.LabelFrame(parent, text=f"Cluster {cluster_id} Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Get cluster data
        center_data = self.processor.dino_data.cluster_centers[cluster_id]
        
        info_text = f"Shapes in cluster: {center_data['num_shapes']}\n"
        # info_text += f"Representative index: {center_data.get('representative_index', 'N/A')}"
        
        info_label = ttk.Label(info_frame, text=info_text, font=('TkDefaultFont', 9))
        info_label.pack(anchor=tk.W)
        
        # NEW: Shape Selection Info
        selection_info_frame = ttk.LabelFrame(parent, text="Shape Selection", padding=10)
        selection_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Selected shape display
        selected_shape_var = tk.StringVar(value="No shape selected")
        setattr(self, f'cluster_{cluster_id}_selected_shape', selected_shape_var)
        
        ttk.Label(selection_info_frame, text="Selected Shape:").pack(anchor=tk.W)
        selected_shape_label = ttk.Label(selection_info_frame, textvariable=selected_shape_var, 
                                    font=('TkDefaultFont', 9, 'bold'), foreground='blue')
        selected_shape_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Selection controls
        selection_controls_frame = ttk.Frame(selection_info_frame)
        selection_controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(selection_controls_frame, text="Clear Selection", 
                command=lambda: self.clear_shape_selection(cluster_id)).pack(side=tk.LEFT, padx=2)
        ttk.Button(selection_controls_frame, text="Highlight Selected", 
                command=lambda: self.highlight_selected_shape(cluster_id)).pack(side=tk.LEFT, padx=2)
        
        # Enable/disable selection mode
        selection_mode_var = tk.BooleanVar(value=True)
        setattr(self, f'cluster_{cluster_id}_selection_mode', selection_mode_var)
        
        ttk.Checkbutton(selection_info_frame, text="Enable shape selection (click to select)", 
                    variable=selection_mode_var,
                    command=lambda: self.toggle_selection_mode(cluster_id)).pack(anchor=tk.W, pady=5)
        
        # View controls
        view_frame = ttk.LabelFrame(parent, text="View Options", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        # View mode for this cluster
        cluster_view_var = tk.StringVar(value="center_only")
        setattr(self, f'cluster_{cluster_id}_view_var', cluster_view_var)
        
        ttk.Radiobutton(view_frame, text="Center Only", variable=cluster_view_var, value="center_only", 
                    command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="All Shapes in Cluster", variable=cluster_view_var, value="all_shapes", 
                    command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
        
        # Rotation controls specific to this cluster
        rotation_frame = ttk.LabelFrame(parent, text="Center Rotation Controls", padding=10)
        rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Rotation angle for this cluster
        ttk.Label(rotation_frame, text="Rotation Angle:").pack(anchor=tk.W)
        cluster_rotation_var = tk.DoubleVar(value=90.0)
        ttk.Scale(rotation_frame, from_=5, to=90, variable=cluster_rotation_var,
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        # Store the rotation variable for this cluster
        setattr(self, f'cluster_{cluster_id}_rotation_var', cluster_rotation_var)
        
        # Rotation buttons
        rot_buttons_frame = ttk.Frame(rotation_frame)
        rot_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(rot_buttons_frame, text="Rotate X", 
                command=lambda: self.rotate_cluster_in_tab(cluster_id, 'x')).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_buttons_frame, text="Rotate Y", 
                command=lambda: self.rotate_cluster_in_tab(cluster_id, 'y')).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_buttons_frame, text="Rotate Z", 
                command=lambda: self.rotate_cluster_in_tab(cluster_id, 'z')).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(rotation_frame, text="Reset Center Rotation", 
                command=lambda: self.reset_cluster_rotation_in_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
        # NEW: Selected Shape Rotation Controls
        selected_rotation_frame = ttk.LabelFrame(parent, text="Selected Shape Controls", padding=10)
        selected_rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(selected_rotation_frame, text="Rotate Selected Shape:").pack(anchor=tk.W)
        selected_rot_buttons_frame = ttk.Frame(selected_rotation_frame)
        selected_rot_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(selected_rot_buttons_frame, text="Rotate X", 
                command=lambda: self.rotate_selected_shape(cluster_id, 'x')).pack(side=tk.LEFT, padx=2)
        ttk.Button(selected_rot_buttons_frame, text="Rotate Y", 
                command=lambda: self.rotate_selected_shape(cluster_id, 'y')).pack(side=tk.LEFT, padx=2)
        ttk.Button(selected_rot_buttons_frame, text="Rotate Z", 
                command=lambda: self.rotate_selected_shape(cluster_id, 'z')).pack(side=tk.LEFT, padx=2)
        
        # ttk.Button(selected_rotation_frame, text="Reset Selected Shape", 
        #         command=lambda: self.reset_selected_shape_rotation(cluster_id)).pack(fill=tk.X, pady=2)
        
        # Canonicalization controls
        canonicalization_frame = ttk.LabelFrame(parent, text="Cluster Canonicalization", padding=10)
        canonicalization_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(canonicalization_frame, text="Align all shapes to cluster center:", 
                font=('TkDefaultFont', 9)).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Button(canonicalization_frame, text="Canonicalize Cluster", 
                command=lambda: self.canonicalize_cluster_shapes(cluster_id)).pack(fill=tk.X, pady=2)
        ttk.Button(canonicalization_frame, text="Reset Shape Alignments", 
                command=lambda: self.reset_cluster_canonicalization(cluster_id)).pack(fill=tk.X, pady=2)
        
        # Progress indicator for canonicalization
        canonicalization_progress_var = tk.StringVar(value="Ready")
        setattr(self, f'cluster_{cluster_id}_canonicalization_status', canonicalization_progress_var)
        
        progress_label = ttk.Label(canonicalization_frame, textvariable=canonicalization_progress_var, 
                                font=('TkDefaultFont', 8), foreground='blue')
        progress_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Actions
        action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Export Cluster", 
                command=lambda: self.export_cluster(cluster_id)).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Close Tab", 
                command=lambda: self.close_cluster_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
        # Status for this cluster
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        cluster_status_text = tk.Text(status_frame, height=6, width=25)
        cluster_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=cluster_status_text.yview)
        cluster_status_text.configure(yscrollcommand=cluster_scrollbar.set)
        cluster_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cluster_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store status text widget for this cluster
        setattr(self, f'cluster_{cluster_id}_status_text', cluster_status_text)


        debug_frame = ttk.LabelFrame(parent, text="Debug Tools", padding=10)
        debug_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(debug_frame, text="Debug Selection Grid", 
                command=lambda: self.debug_coordinate_systems(cluster_id)).pack(side=tk.LEFT, padx=2)
        # ttk.Button(debug_frame, text="Debug Visualization Grid", 
        #         command=lambda: self.debug_visualization_grid(cluster_id)).pack(side=tk.LEFT, padx=2)
        # ttk.Button(debug_frame, text="Compare Grids", 
        #         command=lambda: self.compare_grids(cluster_id)).pack(side=tk.LEFT, padx=2)
        


    def compute_mutual_correspondences_matmul_batch(self, kpts_features1_batch, kpts_features2_batch):
        """
        Compute mutual correspondences between two sets of keypoint features in batch using matrix multiplication.
        This method computes cosine similarity (assuming normalized features) via matmul for each batch.

        Parameters:
            kpts_features1_batch (torch.Tensor): Feature descriptors of shape (B, K1, D)
            kpts_features2_batch (torch.Tensor): Feature descriptors of shape (B, K2, D)
        
        Returns:
            mutual_indices_A (list of torch.Tensor): Indices in each batch of A that have mutual correspondences.
            mutual_indices_B (list of torch.Tensor): Corresponding indices in each batch of B.
        """
        # Normalize the features to unit norm (important for cosine similarity)
        feats_A = kpts_features1_batch / (kpts_features1_batch.norm(dim=2, keepdim=True) + 1e-8)
        feats_B = kpts_features2_batch / (kpts_features2_batch.norm(dim=2, keepdim=True) + 1e-8)
        
        # Compute similarity matrix using matmul; result shape: (B, K1, K2)
        similarity = torch.bmm(feats_A, feats_B.transpose(1, 2))
        
        # Set similarity threshold
        similarity[similarity < 0.1] = 0

        # For each batch, find the highest similarity
        indices_A_to_B = torch.argmax(similarity, dim=2)  # Shape: (B, K1)
        indices_B_to_A = torch.argmax(similarity, dim=1)  # Shape: (B, K2)

        mutual_indices_A = []
        mutual_indices_B = []
        
        # Process each batch independently
        for i in range(kpts_features1_batch.shape[0]):
            batch_indices_A_to_B = indices_A_to_B[i]
            batch_indices_B_to_A = indices_B_to_A[i]
            
            batch_mutual_indices_A = []
            batch_mutual_indices_B = []
            
            for j, b_idx in enumerate(batch_indices_A_to_B):
                sim_val = similarity[i, j, b_idx]
                if sim_val == 0:
                    continue
                
                batch_mutual_indices_A.append(j)
                batch_mutual_indices_B.append(b_idx.item())
            
            mutual_indices_A.append(torch.tensor(batch_mutual_indices_A))
            mutual_indices_B.append(torch.tensor(batch_mutual_indices_B))
        
        return mutual_indices_A, mutual_indices_B

    def rotate_pointcloud_z(self, pointcloud, angle_degrees):
        """
        Rotate a point cloud around the z-axis by a given angle (in degrees) using PyTorch tensors.
        
        Args:
            pointcloud (torch.Tensor): (N, 3) tensor of points.
            angle_degrees (float): Rotation angle in degrees.
        
        Returns:
            torch.Tensor: Rotated point cloud.
        """
        import math
        angle_rad = math.radians(angle_degrees)
        R = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                        [math.sin(angle_rad),  math.cos(angle_rad), 0],
                        [0,                    0,                   1]],
                        dtype=pointcloud.dtype, device=pointcloud.device)
        
        return pointcloud @ R.T

    def positional_encoding_3d(self, xyz, num_frequencies=8):
        """
        Computes a 3D positional encoding with 384 dimensions.
        Args:
            xyz: Tensor of shape (N, 3), where N is the number of points.
            num_frequencies: Number of frequency bands (default: 8, leading to 384 dimensions).
        Returns:
            Tensor of shape (N, 384).
        """
        assert xyz.shape[1] == 3, "Input must have shape (N, 3)"
        
        # Generate frequency bands (log space)
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        
        # Apply positional encoding
        xyz_expanded = xyz[:, None, :] * freq_bands[None, :, None]  # (N, num_frequencies, 3)
        encoding = torch.cat([torch.sin(xyz_expanded), torch.cos(xyz_expanded)], dim=-1)  # (N, num_frequencies, 6)
        
        return encoding.view(xyz.shape[0], -1)  # (N, 384)

    def compute_relative_rotation(self, P, Q):
        """
        Compute the relative rotation matrix between two point clouds given correspondences.

        Parameters:
            P (numpy.ndarray): Nx3 array of source points.
            Q (numpy.ndarray): Nx3 array of target points.

        Returns:
            numpy.ndarray: 3x3 rotation matrix.
        """
        # Center the point clouds
        P_centered = P 
        Q_centered = Q 

        # Compute cross-covariance matrix
        H = P_centered.T @ Q_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure proper rotation (avoid reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        return R

    def compute_z_angle(self, R):
        """
        Extract the rotation angle around the Z-axis (yaw) from a 3x3 rotation matrix.

        Parameters:
            R (numpy.ndarray): 3x3 rotation matrix.

        Returns:
            float: Rotation angle around Z-axis in degrees.
        """
        theta_z = np.arctan2(R[1, 0], R[0, 0])
        return np.degrees(theta_z)

    def canonicalize_cluster_shapes(self, cluster_id):
        """
        Canonicalize all shapes in a cluster by aligning them to the cluster center shape.
        This method finds the best rotation for each shape to align with the center.
        """
        if (self.processor.dino_data is None or 
            not hasattr(self.processor.dino_data, 'cluster_centers') or
            cluster_id not in self.processor.dino_data.cluster_centers):
            self.log_cluster_status(cluster_id, "Error: Cluster center not found")
            return

        # Get cluster data
        center_data = self.processor.dino_data.cluster_centers[cluster_id]
        cluster_labels = self.processor.dino_data.cluster_labels
        cluster_mask = cluster_labels == cluster_id
        
        if not cluster_mask.any():
            self.log_cluster_status(cluster_id, "Error: No shapes found in cluster")
            return

        # Get all shapes in this cluster
        cluster_shape_indices = np.where(cluster_mask)[0]
        cluster_shapes = self.processor.dino_data.kpts_data[cluster_mask]
        
        # Get the center shape (with current rotation applied)
        center_shape = self.apply_rotation_to_cluster_center(center_data['center_shape'], cluster_id)
        representative_idx = center_data.get('representative_index', -1)
        
        self.log_cluster_status(cluster_id, f"Starting canonicalization for {len(cluster_shapes)} shapes...")
        
        # Store original shapes and computed rotations
        if not hasattr(self.processor.dino_data, 'cluster_shape_rotations'):
            self.processor.dino_data.cluster_shape_rotations = {}
        
        if cluster_id not in self.processor.dino_data.cluster_shape_rotations:
            self.processor.dino_data.cluster_shape_rotations[cluster_id] = {}
        
        rotation_angles = {}
        processed_count = 0
        
        # Process each shape in the cluster
        for i, shape_idx in enumerate(cluster_shape_indices):
            if shape_idx == representative_idx:
                # Skip the representative shape (it's the center)
                rotation_angles[shape_idx] = 0.0
                self.processor.dino_data.cluster_shape_rotations[cluster_id][shape_idx] = 0.0
                continue
                
            try:
                # Get current shape
                current_shape = cluster_shapes[i]  # Shape: (N_points, 387)
                
                # Prepare center and current shape data
                center_data_tensor = torch.from_numpy(center_shape).float()
                current_data_tensor = torch.from_numpy(current_shape).float()
                
                # Test different rotation angles
                rotate_angle_list = [0, 90, 180, -90]
                
                query_data_batch = []
                center_data_batch = []
                
                # Apply PCA to both shapes for better feature representation
                from sklearn.decomposition import PCA
                
                for angle in rotate_angle_list:
                    # Rotate the current shape
                    rotated_current = current_data_tensor.clone()
                    rotated_current[:, :3] = self.rotate_pointcloud_z(rotated_current[:, :3], angle)
                    
                    # Apply co-PCA on current and center features
                    pca = PCA(n_components=64)
                    combined_features = np.concatenate([rotated_current[:, 3:], center_data_tensor[:, 3:]], axis=0)
                    pca.fit(combined_features)
                    
                    current_pca = torch.from_numpy(pca.transform(rotated_current[:, 3:]))
                    center_pca = torch.from_numpy(pca.transform(center_data_tensor[:, 3:]))
                    
                    # Add positional encoding
                    current_pos_enc = self.positional_encoding_3d(rotated_current[:, :3])
                    center_pos_enc = self.positional_encoding_3d(center_data_tensor[:, :3])
                    
                    # Combine features
                    current_enhanced = torch.cat([rotated_current[:, :3], current_pca, current_pos_enc], dim=-1)
                    center_enhanced = torch.cat([center_data_tensor[:, :3], center_pca, center_pos_enc], dim=-1)
                    
                    query_data_batch.append(current_enhanced)
                    center_data_batch.append(center_enhanced)
                
                # Stack batches
                query_data_batch = torch.stack(query_data_batch)
                center_data_batch = torch.stack(center_data_batch)
                
                # Compute correspondences for all rotation candidates
                idx_query_batch, idx_center_batch = self.compute_mutual_correspondences_matmul_batch(
                    query_data_batch[:, :, 3:], center_data_batch[:, :, 3:]
                )
                
                # Evaluate each rotation candidate
                error_list = []
                for j in range(len(idx_query_batch)):
                    if len(idx_query_batch[j]) < 10:  # Need sufficient correspondences
                        error_list.append(float('inf'))
                        continue
                    
                    # Get corresponding points
                    query_points = query_data_batch[j][idx_query_batch[j]][:, :3].numpy()
                    center_points = center_data_batch[j][idx_center_batch[j]][:, :3].numpy()
                    
                    # Compute relative rotation and angle error
                    try:
                        rot = self.compute_relative_rotation(query_points, center_points)
                        z_angle = self.compute_z_angle(rot)
                        error = np.abs(z_angle)
                    except:
                        error = float('inf')
                    
                    error_list.append(error)
                
                # Find the best rotation
                best_match = np.argmin(error_list)
                best_angle = rotate_angle_list[best_match]
                best_error = error_list[best_match]
                
                # Store the rotation
                rotation_angles[shape_idx] = best_angle
                self.processor.dino_data.cluster_shape_rotations[cluster_id][shape_idx] = best_angle
                
                # Apply the rotation to the original data
                original_points = self.processor.dino_data.kpts_data[shape_idx, :, :3]
                rotated_points = self.rotate_pointcloud_z(torch.from_numpy(original_points), best_angle).numpy()
                self.processor.dino_data.kpts_data[shape_idx, :, :3] = rotated_points
                
                processed_count += 1
                self.log_cluster_status(cluster_id, f"Shape {shape_idx}: rotated by {best_angle}° (error: {best_error:.2f})")
                
            except Exception as e:
                self.log_cluster_status(cluster_id, f"Error processing shape {shape_idx}: {str(e)}")
                rotation_angles[shape_idx] = 0.0
                self.processor.dino_data.cluster_shape_rotations[cluster_id][shape_idx] = 0.0
        
        # Update visualization
        self.update_cluster_tab_visualization(cluster_id)

        # fine_tune the rotation result
        
        # Log completion
        self.log_cluster_status(cluster_id, f"Canonicalization complete! Processed {processed_count} shapes.")
        self.log_cluster_status(cluster_id, f"Rotation summary: {rotation_angles}")
        
        # Update main view
        if hasattr(self, 'viewer_update'):
            self.viewer_update('tsne')

    def reset_cluster_canonicalization(self, cluster_id):
        """
        Reset all shape rotations in the cluster to their original orientations.
        """
        if (self.processor.dino_data is None or 
            not hasattr(self.processor.dino_data, 'cluster_centers') or
            cluster_id not in self.processor.dino_data.cluster_centers):
            return

        # Reset shape rotations if they exist
        if (hasattr(self.processor.dino_data, 'cluster_shape_rotations') and
            cluster_id in self.processor.dino_data.cluster_shape_rotations):
            
            shape_rotations = self.processor.dino_data.cluster_shape_rotations[cluster_id]
            
            for shape_idx, angle in shape_rotations.items():
                if angle != 0:
                    # Reverse the rotation
                    original_points = self.processor.dino_data.kpts_data[shape_idx, :, :3]
                    reset_points = self.rotate_pointcloud_z(torch.from_numpy(original_points), -angle).numpy()
                    self.processor.dino_data.kpts_data[shape_idx, :, :3] = reset_points
            
            # Clear the rotation records
            self.processor.dino_data.cluster_shape_rotations[cluster_id] = {}
            
            self.log_cluster_status(cluster_id, "All shape rotations reset to original orientations")
        
        # Update visualization
        self.update_cluster_tab_visualization(cluster_id)
        
        # Update main view
        if hasattr(self, 'viewer_update'):
            self.viewer_update('tsne')

    # Modify the setup_cluster_controls method to add canonicalization buttons
    def setup_cluster_controls_updated(self, parent, cluster_id):
        """
        Updated setup controls specific to a cluster tab - includes canonicalization functionality
        """
        # Cluster info
        info_frame = ttk.LabelFrame(parent, text=f"Cluster {cluster_id} Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Get cluster data
        center_data = self.processor.dino_data.cluster_centers[cluster_id]
        
        info_text = f"Shapes in cluster \n: {center_data['num_shapes']}\n"
        # info_text += f"Representative index: {center_data.get('representative_index', 'N/A')}"
        
        info_label = ttk.Label(info_frame, text=info_text, font=('TkDefaultFont', 9))
        info_label.pack(anchor=tk.W)
        
        # View controls
        view_frame = ttk.LabelFrame(parent, text="View Options", padding=10)
        view_frame.pack(fill=tk.X, pady=(0, 10))
        
        # View mode for this cluster
        cluster_view_var = tk.StringVar(value="center_only")
        setattr(self, f'cluster_{cluster_id}_view_var', cluster_view_var)
        
        ttk.Radiobutton(view_frame, text="Center Only", variable=cluster_view_var, value="center_only", 
                    command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
        ttk.Radiobutton(view_frame, text="All Shapes in Cluster", variable=cluster_view_var, value="all_shapes", 
                    command=lambda: self.update_cluster_tab_visualization(cluster_id)).pack(anchor=tk.W)
        
        # Rotation controls specific to this cluster
        rotation_frame = ttk.LabelFrame(parent, text="Center Rotation Controls", padding=10)
        rotation_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Rotation angle for this cluster
        ttk.Label(rotation_frame, text="Rotation Angle:").pack(anchor=tk.W)
        cluster_rotation_var = tk.DoubleVar(value=90.0)
        ttk.Scale(rotation_frame, from_=5, to=90, variable=cluster_rotation_var,
                orient=tk.HORIZONTAL).pack(fill=tk.X, pady=2)
        
        # Store the rotation variable for this cluster
        setattr(self, f'cluster_{cluster_id}_rotation_var', cluster_rotation_var)
        
        # Rotation buttons
        rot_buttons_frame = ttk.Frame(rotation_frame)
        rot_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(rot_buttons_frame, text="Rotate X", 
                command=lambda: self.rotate_cluster_in_tab(cluster_id, 'x')).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_buttons_frame, text="Rotate Y", 
                command=lambda: self.rotate_cluster_in_tab(cluster_id, 'y')).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_buttons_frame, text="Rotate Z", 
                command=lambda: self.rotate_cluster_in_tab(cluster_id, 'z')).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(rotation_frame, text="Reset Center Rotation", 
                command=lambda: self.reset_cluster_rotation_in_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
        # NEW: Canonicalization controls
        canonicalization_frame = ttk.LabelFrame(parent, text="Cluster Canonicalization", padding=10)
        canonicalization_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(canonicalization_frame, text="Align all shapes to cluster center:", 
                font=('TkDefaultFont', 9)).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Button(canonicalization_frame, text="Canonicalize Cluster", 
                command=lambda: self.canonicalize_cluster_shapes(cluster_id)).pack(fill=tk.X, pady=2)
        ttk.Button(canonicalization_frame, text="Reset Shape Alignments", 
                command=lambda: self.reset_cluster_canonicalization(cluster_id)).pack(fill=tk.X, pady=2)
        
        # Progress indicator for canonicalization
        canonicalization_progress_var = tk.StringVar(value="Ready")
        setattr(self, f'cluster_{cluster_id}_canonicalization_status', canonicalization_progress_var)
        
        progress_label = ttk.Label(canonicalization_frame, textvariable=canonicalization_progress_var, 
                                font=('TkDefaultFont', 8), foreground='blue')
        progress_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Actions
        action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Export Cluster", command=lambda: self.export_cluster(cluster_id)).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Close Tab",  command=lambda: self.close_cluster_tab(cluster_id)).pack(fill=tk.X, pady=2)
        
        # Status for this cluster
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        cluster_status_text = tk.Text(status_frame, height=6, width=25)
        cluster_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=cluster_status_text.yview)
        cluster_status_text.configure(yscrollcommand=cluster_scrollbar.set)
        cluster_status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cluster_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store status text widget for this cluster
        setattr(self, f'cluster_{cluster_id}_status_text', cluster_status_text)




    













    # def setup_cluster_visualization(self, parent, cluster_id):
    #     """Setup 3D visualization for a specific cluster"""
    #     # Create matplotlib 3D figure for this cluster
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d', position=[0, 0, 1, 1])
    #     fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
    #     # Embed in tkinter
    #     canvas = FigureCanvasTkAgg(fig, parent)
    #     canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
    #     # Add toolbar
    #     toolbar_frame = ttk.Frame(parent)
    #     toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    #     toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    #     toolbar.update()
        
    #     # Setup axis
    #     ax.set_axis_off()
    #     ax.grid(False)
    #     ax.set_title(f'Cluster {cluster_id} Visualization')
        
    #     return fig, ax, canvas, toolbar


    def setup_cluster_visualization(self, parent, cluster_id):
        """Setup 3D visualization for a specific cluster with mouse selection"""
        return self.setup_cluster_visualization_with_selection(parent, cluster_id)



    def setup_cluster_visualization_with_selection(self, parent, cluster_id):
        """Setup 3D visualization for a specific cluster with mouse selection"""
        # Create matplotlib 3D figure for this cluster
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d', position=[0, 0, 1, 1])
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # Setup axis
        ax.set_axis_off()
        ax.grid(False)
        ax.set_title(f'Cluster {cluster_id} Visualization')
        
        # CRITICAL: Connect mouse click event
        canvas.mpl_connect('button_press_event', lambda event: self.on_cluster_canvas_click(event, cluster_id))
        
        return fig, ax, canvas, toolbar

    def on_cluster_canvas_click(self, event, cluster_id):
        """Handle mouse click on cluster visualization"""
        # Check if selection mode is enabled
        selection_mode_var = getattr(self, f'cluster_{cluster_id}_selection_mode', None)
        if not selection_mode_var or not selection_mode_var.get():
            return
        
        # Only handle left clicks inside the plot
        if event.inaxes is None or event.button != 1:
            return
        
        try:
            # Get the axes for this cluster
            if cluster_id not in self.cluster_tabs:
                return
            
            ax = self.cluster_tabs[cluster_id]['ax']
            
            # Get click coordinates (2D only, no zdata)
            click_x, click_y = event.xdata, event.ydata
            if click_x is None or click_y is None:
                return
            
            self.log_cluster_status(cluster_id, f"Click detected at ({click_x:.2f}, {click_y:.2f})")
            
            # Find the closest shape to the click (remove zdata parameter)
            selected_shape_idx = self.find_closest_shape_to_click(cluster_id, click_x, click_y)
            
            if selected_shape_idx is not None:
                self.select_shape_in_cluster(cluster_id, selected_shape_idx)
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error in click handling: {str(e)}")

    def find_closest_shape_to_click(self, cluster_id, click_x, click_y):
        """Find the shape closest to the mouse click coordinates (2D only)"""
        try:
            # Get cluster data
            cluster_labels = self.processor.dino_data.cluster_labels
            cluster_mask = cluster_labels == cluster_id
            cluster_shape_indices = np.where(cluster_mask)[0]
            
            if len(cluster_shape_indices) == 0:
                return None
            
            # Get view mode
            view_var = getattr(self, f'cluster_{cluster_id}_view_var', None)
            view_mode = view_var.get() if view_var else "center_only"
            
            if view_mode == "center_only":
                # Only center shape is displayed
                center_data = self.processor.dino_data.cluster_centers[cluster_id]
                return center_data.get('representative_index', cluster_shape_indices[0])
            
            elif view_mode == "all_shapes":
                # Multiple shapes in grid layout
                return self.find_closest_shape_in_grid(cluster_id, click_x, click_y, cluster_shape_indices)
            
            return None
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error finding closest shape: {str(e)}")
            return None

    def find_closest_shape_in_grid(self, cluster_id, click_x, click_y, cluster_shape_indices):
        """Find closest shape - FIXED for coordinate system mismatch"""
        return self.find_closest_shape_with_3d_projection(cluster_id, click_x, click_y, cluster_shape_indices)


    def find_closest_shape_with_proper_coordinates(self, cluster_id, click_x, click_y, cluster_shape_indices):
        """Find closest shape using proper coordinate handling"""
        try:
            n_shapes = len(cluster_shape_indices)
            
            # Create exact grid as in visualization
            cols = int(np.sqrt(n_shapes))
            if cols * cols < n_shapes:
                cols += 1
            
            spacing = 1.5
            grid_centers = create_array(spacing, cols, cols)[:n_shapes]
            
            self.log_cluster_status(cluster_id, f"Grid positions range:")
            self.log_cluster_status(cluster_id, f"  X: {grid_centers[:, 0].min():.1f} to {grid_centers[:, 0].max():.1f}")
            self.log_cluster_status(cluster_id, f"  Y: {grid_centers[:, 1].min():.1f} to {grid_centers[:, 1].max():.1f}")
            self.log_cluster_status(cluster_id, f"Click at: ({click_x:.3f}, {click_y:.3f})")
            
            # Check if click is anywhere near the grid
            x_near_grid = abs(click_x - grid_centers[:, 0].mean()) < 10  # Within 10 units
            y_near_grid = abs(click_y - grid_centers[:, 1].mean()) < 10  # Within 10 units
            
            if not (x_near_grid and y_near_grid):
                self.log_cluster_status(cluster_id, 
                    f"Click ({click_x:.3f}, {click_y:.3f}) is very far from grid center ({grid_centers[:, 0].mean():.1f}, {grid_centers[:, 1].mean():.1f})")
                self.log_cluster_status(cluster_id, "This suggests a coordinate system issue!")
            
            # Find closest grid center
            distances = []
            for i, center in enumerate(grid_centers):
                dist = np.sqrt((center[0] - click_x)**2 + (center[1] - click_y)**2)
                distances.append(dist)
                
                # Log all distances for debugging
                self.log_cluster_status(cluster_id, 
                    f"Grid {i}: ({center[0]:.1f}, {center[1]:.1f}) distance: {dist:.3f}")
            
            closest_grid_idx = np.argmin(distances)
            min_distance = distances[closest_grid_idx]
            
            # Use a much more reasonable threshold
            # Since grid spacing is 1.5, half of that (0.75) is reasonable
            reasonable_threshold = spacing * 0.5  # 0.75 for spacing=1.5
            
            self.log_cluster_status(cluster_id, 
                f"Closest: Grid {closest_grid_idx} at distance {min_distance:.3f}, threshold: {reasonable_threshold:.3f}")
            
            if min_distance > reasonable_threshold:
                self.log_cluster_status(cluster_id, 
                    f"Distance {min_distance:.3f} > threshold {reasonable_threshold:.3f}, no selection")
                
                # SPECIAL DEBUG: Try to understand the coordinate transformation
                ax = self.cluster_tabs[cluster_id]['ax']
                
                # Try different coordinate transformations
                try:
                    # Method 1: Check if we need to transform coordinates
                    from mpl_toolkits.mplot3d import proj3d
                    
                    # Get the view transformation matrix
                    M = ax.get_proj()
                    
                    # Try to reverse-transform a grid position to see what click coordinates should be
                    test_grid_pos = grid_centers[0]  # First grid position
                    
                    # Transform grid position to screen coordinates, then back
                    x2d, y2d, _ = proj3d.proj_transform(test_grid_pos[0], test_grid_pos[1], test_grid_pos[2], M)
                    
                    self.log_cluster_status(cluster_id, 
                        f"TRANSFORM TEST: Grid ({test_grid_pos[0]:.1f}, {test_grid_pos[1]:.1f}, {test_grid_pos[2]:.1f}) -> Screen ({x2d:.3f}, {y2d:.3f})")
                    self.log_cluster_status(cluster_id, 
                        f"Your clicks are around ({click_x:.3f}, {click_y:.3f}) - much closer to screen coordinates!")
                    
                except Exception as transform_error:
                    self.log_cluster_status(cluster_id, f"Transform test error: {transform_error}")
                
                return None
            
            if closest_grid_idx < len(cluster_shape_indices):
                return cluster_shape_indices[closest_grid_idx]
            
            return None
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error in coordinate-aware search: {str(e)}")
            return None

    def find_closest_shape_with_3d_projection(self, cluster_id, click_x, click_y, cluster_shape_indices):
        """Find closest shape using 3D to 2D projection - CORRECT METHOD"""
        try:
            n_shapes = len(cluster_shape_indices)
            
            # Create exact grid as in visualization
            cols = int(np.sqrt(n_shapes))
            if cols * cols < n_shapes:
                cols += 1
            
            spacing = 1.5
            grid_centers = create_array(spacing, cols, cols)[:n_shapes]
            
            # Get the 3D axis
            ax = self.cluster_tabs[cluster_id]['ax']
            
            # Project each 3D grid center to 2D screen coordinates
            from mpl_toolkits.mplot3d import proj3d
            
            projected_centers = []
            for center in grid_centers:
                # Project 3D world coordinates to 2D screen coordinates
                x2d, y2d, _ = proj3d.proj_transform(center[0], center[1], center[2], ax.get_proj())
                projected_centers.append([x2d, y2d])
            
            projected_centers = np.array(projected_centers)
            
            self.log_cluster_status(cluster_id, f"PROJECTION METHOD:")
            self.log_cluster_status(cluster_id, f"Click: ({click_x:.3f}, {click_y:.3f})")
            
            # Find closest projected center
            distances = []
            for i, proj_center in enumerate(projected_centers):
                dist = np.sqrt((proj_center[0] - click_x)**2 + (proj_center[1] - click_y)**2)
                distances.append(dist)
                
                # Log first few for debugging
                if i < 3:
                    original_pos = grid_centers[i]
                    self.log_cluster_status(cluster_id, 
                        f"Grid {i}: 3D({original_pos[0]:.1f}, {original_pos[1]:.1f}) -> 2D({proj_center[0]:.3f}, {proj_center[1]:.3f}) dist: {dist:.3f}")
            
            closest_grid_idx = np.argmin(distances)
            min_distance = distances[closest_grid_idx]
            
            # Use a reasonable threshold in screen coordinate space
            # Screen coordinates are usually in range [-1, 1] or similar
            screen_threshold = 0.2  # Adjust based on testing
            
            self.log_cluster_status(cluster_id, 
                f"PROJECTION: Closest Grid {closest_grid_idx}, distance: {min_distance:.3f}, threshold: {screen_threshold}")
            
            if min_distance > screen_threshold:
                self.log_cluster_status(cluster_id, "Click too far in screen coordinates")
                return None
            
            if closest_grid_idx < len(cluster_shape_indices):
                closest_shape_idx = cluster_shape_indices[closest_grid_idx]
                self.log_cluster_status(cluster_id, f"SELECTED: Shape {closest_shape_idx}")
                return closest_shape_idx
            
            return None
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error in projection method: {str(e)}")
            # Fallback to coordinate-aware method
            return self.find_closest_shape_with_proper_coordinates(cluster_id, click_x, click_y, cluster_shape_indices)



    # def add_coordinate_debug_button(self, parent, cluster_id):
    #     """Add coordinate system debug button"""
    #     debug_frame = ttk.LabelFrame(parent, text="Coordinate Debug", padding=5)
    #     debug_frame.pack(fill=tk.X, pady=(0, 5))
        
    #     ttk.Button(debug_frame, text="Debug Coordinates", 
    #             command=lambda: self.debug_coordinate_systems(cluster_id)).pack(side=tk.LEFT, padx=2)
        


    def debug_coordinate_systems(self, cluster_id):
        """Debug coordinate systems to understand the mismatch"""
        try:
            if cluster_id not in self.cluster_tabs:
                return
                
            ax = self.cluster_tabs[cluster_id]['ax']
            
            # Get actual axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            
            self.log_cluster_status(cluster_id, f"=== COORDINATE SYSTEM DEBUG ===")
            self.log_cluster_status(cluster_id, f"Axis limits:")
            self.log_cluster_status(cluster_id, f"  X: {xlim[0]:.2f} to {xlim[1]:.2f}")
            self.log_cluster_status(cluster_id, f"  Y: {ylim[0]:.2f} to {ylim[1]:.2f}")
            self.log_cluster_status(cluster_id, f"  Z: {zlim[0]:.2f} to {zlim[1]:.2f}")
            
            # Get grid positions
            cluster_labels = self.processor.dino_data.cluster_labels
            cluster_mask = cluster_labels == cluster_id
            cluster_shape_indices = np.where(cluster_mask)[0]
            n_shapes = len(cluster_shape_indices)
            
            # Create exact grid
            cols = int(np.sqrt(n_shapes))
            if cols * cols < n_shapes:
                cols += 1
            
            spacing = 1.5
            grid_centers = create_array(spacing, cols, cols)[:n_shapes]
            
            self.log_cluster_status(cluster_id, f"Grid positions:")
            for i, pos in enumerate(grid_centers[:5]):  # Show first 5
                self.log_cluster_status(cluster_id, f"  Grid {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            
            # Check if grid positions are within axis limits
            grid_x_range = (grid_centers[:, 0].min(), grid_centers[:, 0].max())
            grid_y_range = (grid_centers[:, 1].min(), grid_centers[:, 1].max())
            
            self.log_cluster_status(cluster_id, f"Grid ranges:")
            self.log_cluster_status(cluster_id, f"  X: {grid_x_range[0]:.1f} to {grid_x_range[1]:.1f}")
            self.log_cluster_status(cluster_id, f"  Y: {grid_y_range[0]:.1f} to {grid_y_range[1]:.1f}")
            
            # Check if grid is within axis bounds
            x_in_bounds = xlim[0] <= grid_x_range[0] and grid_x_range[1] <= xlim[1]
            y_in_bounds = ylim[0] <= grid_y_range[0] and grid_y_range[1] <= ylim[1]
            
            self.log_cluster_status(cluster_id, f"Grid within axis bounds: X={x_in_bounds}, Y={y_in_bounds}")
            
            if not x_in_bounds or not y_in_bounds:
                self.log_cluster_status(cluster_id, "WARNING: Grid positions are outside axis bounds!")
                self.log_cluster_status(cluster_id, "This explains why clicks near (0,0) don't work!")
            
            self.log_cluster_status(cluster_id, f"=== END COORDINATE DEBUG ===")
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Coordinate debug error: {str(e)}")

    def find_closest_shape_in_grid_fixed(self, cluster_id, click_x, click_y, cluster_shape_indices):
        """Fixed version that matches visualization exactly"""
        try:
            n_shapes = len(cluster_shape_indices)
            
            # Use EXACT same grid as visualization
            spacing = 1.5  # Make sure this matches your visualization
            grid_centers = self.create_exact_visualization_grid(spacing, n_shapes)
            
            self.log_cluster_status(cluster_id, f"Using exact visualization grid with {len(grid_centers)} positions")
            
            # Find closest grid center using simple 2D distance in data coordinates
            distances = []
            for i, center in enumerate(grid_centers):
                dist = np.sqrt((center[0] - click_x)**2 + (center[1] - click_y)**2)
                distances.append(dist)
            
            closest_grid_idx = np.argmin(distances)
            min_distance = distances[closest_grid_idx]
            
            # Get the corresponding shape index
            if closest_grid_idx < len(cluster_shape_indices):
                closest_shape_idx = cluster_shape_indices[closest_grid_idx]
                
                # Log selection details
                self.log_cluster_status(cluster_id, 
                    f"Click ({click_x:.2f}, {click_y:.2f}) -> Grid {closest_grid_idx} -> Shape {closest_shape_idx}")
                self.log_cluster_status(cluster_id, f"Distance: {min_distance:.3f}")
                
                # Reasonable distance threshold
                if min_distance > 2.0:
                    self.log_cluster_status(cluster_id, f"Click too far (distance: {min_distance:.3f})")
                    return None
                
                return closest_shape_idx
            else:
                self.log_cluster_status(cluster_id, f"Invalid grid index: {closest_grid_idx}")
                return None
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error in fixed grid search: {str(e)}")
            return None
    
    
    def create_flexible_grid_for_selection(self, spacing, n_shapes):
        """Create grid for selection - matches visualization exactly"""
        return self.create_exact_visualization_grid(spacing, n_shapes)
    

    def create_exact_visualization_grid(self, spacing, n_shapes):
        """Create the EXACT same grid as used in draw_cluster_all_shapes"""
        if n_shapes == 0:
            return np.array([]).reshape(0, 3)
        
        if n_shapes == 1:
            return np.array([[0, 0, 0]], dtype=np.float32)
        
        # EXACT replication of your draw_cluster_all_shapes grid logic
        cols = int(np.sqrt(n_shapes))
        if cols * cols < n_shapes:
            cols += 1
        
        # CRITICAL: Use the SAME create_array function as your visualization!
        # This is the key difference - your visualization uses create_array(1.5, cols, cols)
        # instead of our manual grid creation
        
        # Replicate create_array function behavior:
        # create_array(x, rows, cols) creates: [[j*x, n*x, 0] for n in range(1, rows+1) for j in range(1, cols+1)]
        
        rows = cols  # In your case, you use cols for both rows and cols in create_array
        grid_positions = []
        
        # This replicates: create_array(1.5, cols, cols)[:n_shapes, None, :].repeat(K, axis=1)
        for n in range(1, rows + 1):  # n from 1 to rows
            for j in range(1, cols + 1):  # j from 1 to cols
                x = j * spacing  # j * x
                y = n * spacing  # n * x  
                z = 0
                grid_positions.append([x, y, z])
                
                if len(grid_positions) >= n_shapes:
                    break
            if len(grid_positions) >= n_shapes:
                break
        
        return np.array(grid_positions[:n_shapes], dtype=np.float32)


    def debug_grid_layout(self, cluster_id):
        """Debug method to check grid layout matches visualization"""
        try:
            if cluster_id not in self.cluster_tabs:
                return
                
            # Get cluster data
            cluster_labels = self.processor.dino_data.cluster_labels
            cluster_mask = cluster_labels == cluster_id
            cluster_shape_indices = np.where(cluster_mask)[0]
            n_shapes = len(cluster_shape_indices)
            
            # Create grid same as selection
            spacing = 1.5
            grid_centers = self.create_flexible_grid_for_selection(spacing, n_shapes)
            
            self.log_cluster_status(cluster_id, f"=== GRID DEBUG ===")
            self.log_cluster_status(cluster_id, f"Number of shapes: {n_shapes}")
            self.log_cluster_status(cluster_id, f"Grid spacing: {spacing}")
            self.log_cluster_status(cluster_id, f"Grid centers shape: {grid_centers.shape}")
            
            for i, (center, shape_idx) in enumerate(zip(grid_centers, cluster_shape_indices)):
                shape_name = self.get_shape_name(shape_idx)
                self.log_cluster_status(cluster_id, 
                    f"Grid {i}: {shape_name} at ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
                
                if i >= 10:  # Limit output
                    self.log_cluster_status(cluster_id, f"... and {n_shapes - i - 1} more")
                    break
                    
            self.log_cluster_status(cluster_id, f"=== END DEBUG ===")
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Debug error: {str(e)}")







    def select_shape_in_cluster(self, cluster_id, shape_idx):
        """Select a shape in the cluster"""
        try:
            # Get shape name
            shape_names_list = getattr(self.processor.dino_data, 'kpts_data_name_list', None)
            if shape_names_list and shape_idx < len(shape_names_list):
                shape_name = shape_names_list[shape_idx]
            else:
                shape_name = f"shape_{shape_idx}"
            
            # Update selected shape variable
            selected_shape_var = getattr(self, f'cluster_{cluster_id}_selected_shape', None)
            if selected_shape_var:
                selected_shape_var.set(f"{shape_name} (index: {shape_idx})")
            
            # Store the selected shape index
            setattr(self, f'cluster_{cluster_id}_selected_shape_idx', shape_idx)
            
            # Update visualization to highlight selected shape
            self.update_cluster_tab_visualization(cluster_id)
            
            self.log_cluster_status(cluster_id, f"Selected shape: {shape_name} (\n index: {shape_idx})")
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error selecting shape: {str(e)}")

    def clear_shape_selection(self, cluster_id):
        """Clear the selected shape"""
        try:
            # Clear selected shape variable
            selected_shape_var = getattr(self, f'cluster_{cluster_id}_selected_shape', None)
            if selected_shape_var:
                selected_shape_var.set("No shape selected")
            
            # Clear stored selection
            if hasattr(self, f'cluster_{cluster_id}_selected_shape_idx'):
                delattr(self, f'cluster_{cluster_id}_selected_shape_idx')
            
            # Update visualization
            self.update_cluster_tab_visualization(cluster_id)
            
            self.log_cluster_status(cluster_id, "Shape selection cleared")
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error clearing selection: {str(e)}")

    def highlight_selected_shape(self, cluster_id):
        """Highlight the selected shape in the visualization"""
        selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
        if selected_shape_idx is None:
            self.log_cluster_status(cluster_id, "No shape selected to highlight")
            return
        
        # Update visualization with highlighting
        self.update_cluster_tab_visualization(cluster_id)
        
        self.log_cluster_status(cluster_id, f"Highlighted selected shape: {selected_shape_idx}")

    def toggle_selection_mode(self, cluster_id):
        """Toggle shape selection mode on/off"""
        selection_mode_var = getattr(self, f'cluster_{cluster_id}_selection_mode', None)
        if selection_mode_var:
            mode = "enabled" if selection_mode_var.get() else "disabled"
            self.log_cluster_status(cluster_id, f"Shape selection mode {mode}")

    def rotate_selected_shape(self, cluster_id, axis):
        """Rotate the selected shape"""
        selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
        if selected_shape_idx is None:
            self.log_cluster_status(cluster_id, "No shape selected for rotation")
            return
        
        # Get rotation angle
        rotation_var = getattr(self, f'cluster_{cluster_id}_rotation_var', None)
        angle = rotation_var.get() if rotation_var else 15.0
        
        # Apply rotation to selected shape
        try:
            original_points = self.processor.dino_data.kpts_data[selected_shape_idx, :, :3]
            rotated_points = self.rotate_pointcloud_axis(torch.from_numpy(original_points), axis, angle).numpy()
            self.processor.dino_data.kpts_data[selected_shape_idx, :, :3] = rotated_points
            
            # Update visualization
            self.update_cluster_tab_visualization(cluster_id)
            
            shape_name = self.get_shape_name(selected_shape_idx)
            self.log_cluster_status(cluster_id, f"Rotated {shape_name} by {angle}° around {axis.upper()}-axis")
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error rotating shape: {str(e)}")

    def rotate_pointcloud_axis(self, pointcloud, axis, angle_degrees):
        """Rotate point cloud around specified axis"""
        angle_rad = np.radians(angle_degrees)
        
        if axis == 'x':
            R = torch.tensor([[1, 0, 0],
                            [0, np.cos(angle_rad), -np.sin(angle_rad)],
                            [0, np.sin(angle_rad), np.cos(angle_rad)]], dtype=torch.float32)
        elif axis == 'y':
            R = torch.tensor([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                            [0, 1, 0],
                            [-np.sin(angle_rad), 0, np.cos(angle_rad)]], dtype=torch.float32)
        elif axis == 'z':
            R = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                            [np.sin(angle_rad), np.cos(angle_rad), 0],
                            [0, 0, 1]], dtype=torch.float32)
        else:
            return pointcloud
        
        return pointcloud @ R.T

    def reset_selected_shape_rotation(self, cluster_id):
        """Reset rotation of selected shape"""
        selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
        if selected_shape_idx is None:
            self.log_cluster_status(cluster_id, "No shape selected for reset")
            return
        
        # This would require storing original coordinates - implement based on your needs
        self.log_cluster_status(cluster_id, "Reset selected shape rotation (implement based on your coordinate storage)")



    def get_shape_name(self, shape_idx):
        """Get the name of a shape by its index"""
        shape_names_list = getattr(self.processor.dino_data, 'kpts_data_name_list', None)
        if shape_names_list and shape_idx < len(shape_names_list):
            return shape_names_list[shape_idx]
        return f"shape_{shape_idx}"

    # Update your existing update_cluster_tab_visualization method to highlight selected shapes
    def update_cluster_tab_visualization_with_selection(self, cluster_id):
        """Update visualization with selection highlighting"""
        if cluster_id not in self.cluster_tabs:
            return
        
        # Get selected shape
        selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
        
        # Call original visualization update
        self.update_cluster_tab_visualization(cluster_id)
        
        # Add selection highlighting if a shape is selected
        if selected_shape_idx is not None:
            self.add_selection_highlighting(cluster_id, selected_shape_idx)

    def add_selection_highlighting(self, cluster_id, selected_shape_idx):
        """Add visual highlighting for selected shape"""
        try:
            ax = self.cluster_tabs[cluster_id]['ax']
            canvas = self.cluster_tabs[cluster_id]['canvas']
            
            # Add a text annotation or bounding box around selected shape
            # This is a simplified version - you can make it more sophisticated
            ax.text2D(0.02, 0.02, f"Selected: {self.get_shape_name(selected_shape_idx)}", 
                    transform=ax.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            canvas.draw_idle()
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Error adding selection highlight: {str(e)}")

    # def setup_cluster_visualization_with_selection(self, parent, cluster_id):
    #     """Setup 3D visualization for a specific cluster with mouse selection"""
    #     # Create matplotlib 3D figure for this cluster
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d', position=[0, 0, 1, 1])
    #     fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        
    #     # Embed in tkinter
    #     canvas = FigureCanvasTkAgg(fig, parent)
    #     canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
    #     # Add toolbar
    #     toolbar_frame = ttk.Frame(parent)
    #     toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    #     toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    #     toolbar.update()
        
    #     # Setup axis
    #     ax.set_axis_off()
    #     ax.grid(False)
    #     ax.set_title(f'Cluster {cluster_id} Visualization')
        
    #     # CRITICAL: Connect mouse click event
    #     canvas.mpl_connect('button_press_event', lambda event: self.on_cluster_canvas_click(event, cluster_id))
        
    #     return fig, ax, canvas, toolbar

    # def on_cluster_canvas_click(self, event, cluster_id):
    #     """Handle mouse click on cluster visualization"""
    #     # Check if selection mode is enabled
    #     selection_mode_var = getattr(self, f'cluster_{cluster_id}_selection_mode', None)
    #     if not selection_mode_var or not selection_mode_var.get():
    #         return
        
    #     # Only handle left clicks inside the plot
    #     if event.inaxes is None or event.button != 1:
    #         return
        
    #     try:
    #         # Get the axes for this cluster
    #         if cluster_id not in self.cluster_tabs:
    #             return
            
    #         ax = self.cluster_tabs[cluster_id]['ax']
            
    #         # Get click coordinates
    #         click_x, click_y = event.xdata, event.ydata
    #         if click_x is None or click_y is None:
    #             return
            
    #         self.log_cluster_status(cluster_id, f"Click detected at ({click_x:.2f}, {click_y:.2f})")
            
    #         # Find the closest shape to the click
    #         selected_shape_idx = self.find_closest_shape_to_click(cluster_id, click_x, click_y, event.zdata)
            
    #         if selected_shape_idx is not None:
    #             self.select_shape_in_cluster(cluster_id, selected_shape_idx)
            
    #     except Exception as e:
    #         self.log_cluster_status(cluster_id, f"Error in click handling: {str(e)}")

    # def find_closest_shape_to_click(self, cluster_id, click_x, click_y, click_z=None):
    #     """Find the shape closest to the mouse click coordinates"""
    #     try:
    #         # Get cluster data
    #         cluster_labels = self.processor.dino_data.cluster_labels
    #         cluster_mask = cluster_labels == cluster_id
    #         cluster_shape_indices = np.where(cluster_mask)[0]
            
    #         if len(cluster_shape_indices) == 0:
    #             return None
            
    #         # Get view mode
    #         view_var = getattr(self, f'cluster_{cluster_id}_view_var', None)
    #         view_mode = view_var.get() if view_var else "center_only"
            
    #         if view_mode == "center_only":
    #             # Only center shape is displayed
    #             center_data = self.processor.dino_data.cluster_centers[cluster_id]
    #             return center_data.get('representative_index', cluster_shape_indices[0])
            
    #         elif view_mode == "all_shapes":
    #             # Multiple shapes in grid layout
    #             return self.find_closest_shape_in_grid(cluster_id, click_x, click_y, cluster_shape_indices)
            
    #         return None
            
    #     except Exception as e:
    #         self.log_cluster_status(cluster_id, f"Error finding closest shape: {str(e)}")
    #         return None

    # def find_closest_shape_in_grid(self, cluster_id, click_x, click_y, cluster_shape_indices):
    #     """Find closest shape in grid layout"""
    #     try:
    #         # Get cluster shapes
    #         cluster_shapes = self.processor.dino_data.kpts_data[cluster_shape_indices]
    #         n_shapes = len(cluster_shapes)
    #         K = cluster_shapes.shape[1]
            
    #         # Recreate the grid layout (same as in draw_cluster_all_shapes)
    #         cols = int(np.sqrt(n_shapes))
    #         if cols * cols < n_shapes:
    #             cols += 1
            
    #         # Create grid centers
    #         grid_centers = []
    #         for i in range(n_shapes):
    #             row = i // cols
    #             col = i % cols
    #             x = col * 1.5  # Same spacing as in visualization
    #             y = row * 1.5
    #             z = 0
    #             grid_centers.append([x, y, z])
            
    #         grid_centers = np.array(grid_centers)
            
    #         # Find closest grid center to click
    #         distances = []
    #         for i, center in enumerate(grid_centers):
    #             # Calculate 2D distance (ignore z for now)
    #             dist = np.sqrt((center[0] - click_x)**2 + (center[1] - click_y)**2)
    #             distances.append(dist)
            
    #         closest_grid_idx = np.argmin(distances)
    #         closest_shape_idx = cluster_shape_indices[closest_grid_idx]
            
    #         self.log_cluster_status(cluster_id, f"Closest shape: grid position {closest_grid_idx}, shape index {closest_shape_idx}")
            
    #         return closest_shape_idx
            
    #     except Exception as e:
    #         self.log_cluster_status(cluster_id, f"Error in grid search: {str(e)}")
    #         return None

    # def select_shape_in_cluster(self, cluster_id, shape_idx):
    #     """Select a shape in the cluster"""
    #     try:
    #         # Get shape name
    #         shape_names_list = getattr(self.processor.dino_data, 'kpts_data_name_list', None)
    #         if shape_names_list and shape_idx < len(shape_names_list):
    #             shape_name = shape_names_list[shape_idx]
    #         else:
    #             shape_name = f"shape_{shape_idx}"
            
    #         # Update selected shape variable
    #         selected_shape_var = getattr(self, f'cluster_{cluster_id}_selected_shape', None)
    #         if selected_shape_var:
    #             selected_shape_var.set(f"{shape_name} (index: {shape_idx})")
            
    #         # Store the selected shape index
    #         setattr(self, f'cluster_{cluster_id}_selected_shape_idx', shape_idx)
            
    #         # Update visualization to highlight selected shape
    #         self.update_cluster_tab_visualization(cluster_id)
            
    #         self.log_cluster_status(cluster_id, f"Selected shape: {shape_name} (index: {shape_idx})")
            
    #     except Exception as e:
    #         self.log_cluster_status(cluster_id, f"Error selecting shape: {str(e)}")

    # def clear_shape_selection(self, cluster_id):
    #     """Clear the selected shape"""
    #     try:
    #         # Clear selected shape variable
    #         selected_shape_var = getattr(self, f'cluster_{cluster_id}_selected_shape', None)
    #         if selected_shape_var:
    #             selected_shape_var.set("No shape selected")
            
    #         # Clear stored selection
    #         if hasattr(self, f'cluster_{cluster_id}_selected_shape_idx'):
    #             delattr(self, f'cluster_{cluster_id}_selected_shape_idx')
            
    #         # Update visualization
    #         self.update_cluster_tab_visualization(cluster_id)
            
    #         self.log_cluster_status(cluster_id, "Shape selection cleared")
            
    #     except Exception as e:
    #         self.log_cluster_status(cluster_id, f"Error clearing selection: {str(e)}")

    # def highlight_selected_shape(self, cluster_id):
    #     """Highlight the selected shape in the visualization"""
    #     selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
    #     if selected_shape_idx is None:
    #         self.log_cluster_status(cluster_id, "No shape selected to highlight")
    #         return
        
    #     # Update visualization with highlighting
    #     self.update_cluster_tab_visualization(cluster_id)
        
    #     self.log_cluster_status(cluster_id, f"Highlighted selected shape: {selected_shape_idx}")

    # def toggle_selection_mode(self, cluster_id):
    #     """Toggle shape selection mode on/off"""
    #     selection_mode_var = getattr(self, f'cluster_{cluster_id}_selection_mode', None)
    #     if selection_mode_var:
    #         mode = "enabled" if selection_mode_var.get() else "disabled"
    #         self.log_cluster_status(cluster_id, f"Shape selection mode {mode}")

    # def rotate_selected_shape(self, cluster_id, axis):
    #     """Rotate the selected shape"""
    #     selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
    #     if selected_shape_idx is None:
    #         self.log_cluster_status(cluster_id, "No shape selected for rotation")
    #         return
        
    #     # Get rotation angle
    #     rotation_var = getattr(self, f'cluster_{cluster_id}_rotation_var', None)
    #     angle = rotation_var.get() if rotation_var else 15.0
        
    #     # Apply rotation to selected shape
    #     try:
    #         original_points = self.processor.dino_data.kpts_data[selected_shape_idx, :, :3]
    #         rotated_points = self.rotate_pointcloud_axis(torch.from_numpy(original_points), axis, angle).numpy()
    #         self.processor.dino_data.kpts_data[selected_shape_idx, :, :3] = rotated_points
            
    #         # Update visualization
    #         self.update_cluster_tab_visualization(cluster_id)
            
    #         shape_name = self.get_shape_name(selected_shape_idx)
    #         self.log_cluster_status(cluster_id, f"Rotated {shape_name} by {angle}° around {axis.upper()}-axis")
            
    #     except Exception as e:
    #         self.log_cluster_status(cluster_id, f"Error rotating shape: {str(e)}")

    # def rotate_pointcloud_axis(self, pointcloud, axis, angle_degrees):
    #     """Rotate point cloud around specified axis"""
    #     angle_rad = np.radians(angle_degrees)
        
    #     if axis == 'x':
    #         R = torch.tensor([[1, 0, 0],
    #                         [0, np.cos(angle_rad), -np.sin(angle_rad)],
    #                         [0, np.sin(angle_rad), np.cos(angle_rad)]], dtype=torch.float32)
    #     elif axis == 'y':
    #         R = torch.tensor([[np.cos(angle_rad), 0, np.sin(angle_rad)],
    #                         [0, 1, 0],
    #                         [-np.sin(angle_rad), 0, np.cos(angle_rad)]], dtype=torch.float32)
    #     elif axis == 'z':
    #         R = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad), 0],
    #                         [np.sin(angle_rad), np.cos(angle_rad), 0],
    #                         [0, 0, 1]], dtype=torch.float32)
    #     else:
    #         return pointcloud
        
    #     return pointcloud @ R.T

    # def reset_selected_shape_rotation(self, cluster_id):
    #     """Reset rotation of selected shape"""
    #     selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
    #     if selected_shape_idx is None:
    #         self.log_cluster_status(cluster_id, "No shape selected for reset")
    #         return
        
    #     # This would require storing original coordinates - implement based on your needs
    #     self.log_cluster_status(cluster_id, "Reset selected shape rotation (implement based on your coordinate storage)")

    # def get_shape_name(self, shape_idx):
    #     """Get the name of a shape by its index"""
    #     shape_names_list = getattr(self.processor.dino_data, 'kpts_data_name_list', None)
    #     if shape_names_list and shape_idx < len(shape_names_list):
    #         return shape_names_list[shape_idx]
    #     return f"shape_{shape_idx}"




    # # Update your existing update_cluster_tab_visualization method to highlight selected shapes
    # def update_cluster_tab_visualization_with_selection(self, cluster_id):
    #     """Update visualization with selection highlighting"""
    #     if cluster_id not in self.cluster_tabs:
    #         return
        
    #     # Get selected shape
    #     selected_shape_idx = getattr(self, f'cluster_{cluster_id}_selected_shape_idx', None)
        
    #     # Call original visualization update
    #     self.update_cluster_tab_visualization(cluster_id)
        
    #     # Add selection highlighting if a shape is selected
    #     if selected_shape_idx is not None:
    #         self.add_selection_highlighting(cluster_id, selected_shape_idx)

    # def add_selection_highlighting(self, cluster_id, selected_shape_idx):
    #     """Add visual highlighting for selected shape"""
    #     try:
    #         ax = self.cluster_tabs[cluster_id]['ax']
    #         canvas = self.cluster_tabs[cluster_id]['canvas']
            
    #         # Add a text annotation or bounding box around selected shape
    #         # This is a simplified version - you can make it more sophisticated
    #         ax.text2D(0.02, 0.02, f"Selected: {self.get_shape_name(selected_shape_idx)}", 
    #                 transform=ax.transAxes, fontsize=10, 
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
    #         canvas.draw_idle()
            
    #     except Exception as e:
    #         self.log_cluster_status(cluster_id, f"Error adding selection highlight: {str(e)}")









    def update_cluster_tab_visualization(self, cluster_id):
        """Update the visualization in a specific cluster tab"""
        if cluster_id not in self.cluster_tabs:
            return
        
        tab_data = self.cluster_tabs[cluster_id]
        ax = tab_data['ax']
        canvas = tab_data['canvas']
        
        # Clear the plot
        ax.clear()
        ax.set_axis_off()
        ax.grid(False)


        
        
        # Get view mode for this cluster
        view_var = getattr(self, f'cluster_{cluster_id}_view_var', None)
        view_mode = view_var.get() if view_var else "center_only"
        
        # Get cluster data
        center_data = self.processor.dino_data.cluster_centers[cluster_id]
        cluster_labels = self.processor.dino_data.cluster_labels
        
        if view_mode == "center_only":
            self.draw_cluster_center_only(ax, cluster_id, center_data)
            
        elif view_mode == "all_shapes":
            self.draw_cluster_all_shapes(ax, cluster_id, center_data, cluster_labels)
            
        
        
        # Add coordinate system
        self.add_coordinate_system_to_ax(ax)
        
        # Set title with rotation info
        rotation_info = self.get_cluster_rotation_info(cluster_id)
        ax.set_title(f'Cluster {cluster_id} - {view_mode.replace("_", " ").title()}\n{rotation_info}')
        
        # Set appropriate bounds
        self.set_cluster_tab_bounds(ax, cluster_id, view_mode)
        
        self.canvas_3d.mpl_connect('scroll_event', self.on_scroll)
        canvas.draw_idle()


    def draw_cluster_center_only(self, ax, cluster_id, center_data):
        """Draw only the cluster center shape"""
        center_shape = center_data['center_shape']
        
        # Apply rotation
        rotated_center_shape = self.apply_rotation_to_cluster_center(center_shape, cluster_id)
        points = rotated_center_shape[:, :3]
        
        # Center the shape at origin
        points_centered = points - np.mean(points, axis=0)
        
        # Get colors
        try:
            colors = apply_pca_and_store_colors(rotated_center_shape.reshape(1, -1, rotated_center_shape.shape[-1]), True)[0]
        except:
            colors = np.tile([0.5, 0.7, 0.9], (len(points_centered), 1))
        
        # Plot
        ax.scatter(points_centered[:, 0], points_centered[:, 1], points_centered[:, 2],
                c=colors, s=4, alpha=0.8)


    def draw_cluster_all_shapes(self, ax, cluster_id, center_data, cluster_labels):
        """Draw all shapes in the cluster with bounding box around rotated cluster center"""
        # Get all shapes in this cluster
        cluster_mask = cluster_labels == cluster_id
        
        if hasattr(self.processor.dino_data, 'scaled_shapes'):
            cluster_shapes = self.processor.dino_data.scaled_shapes[cluster_mask]
        else:
            cluster_shapes = self.processor.dino_data.kpts_data[cluster_mask]
        
        n_shapes = len(cluster_shapes)
        K = cluster_shapes.shape[1]
        
        # Get cluster center information
        center_shape = center_data['center_shape']
        representative_idx = center_data.get('representative_index', -1)
        
        # Find which position in the grid corresponds to the center shape
        cluster_indices = np.where(cluster_mask)[0]
        center_position_in_grid = None
        
        for i, global_idx in enumerate(cluster_indices):
            if global_idx == representative_idx:
                center_position_in_grid = i
                break
        
        # If representative not found, use the first shape as center reference
        if center_position_in_grid is None:
            center_position_in_grid = 0
        
        # Create grid layout for multiple shapes - ALL IN ORIGINAL ORIENTATIONS FIRST
        cols = int(np.sqrt(n_shapes))
        if cols * cols < n_shapes:
            cols += 1
        
        grid_centers = create_array(1.5, cols, cols)[:n_shapes, None, :].repeat(K, axis=1)
        all_points = cluster_shapes[:, :, :3].reshape(-1, 3) + grid_centers.reshape(-1, 3)
        
        # Apply rotation ONLY to the cluster center shape
        if (center_position_in_grid is not None and 
            hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
            cluster_id in self.processor.dino_data.cluster_center_rotations):
            
            rotation_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
            
            if not np.allclose(rotation_matrix, np.eye(3)):
                # Apply rotation ONLY to center shape
                center_start_idx = center_position_in_grid * K
                center_end_idx = (center_position_in_grid + 1) * K
                center_shape_points = all_points[center_start_idx:center_end_idx]
                
                # Rotate center shape around its own center
                shape_center = np.mean(center_shape_points, axis=0)
                centered_points = center_shape_points - shape_center
                rotated_points = np.dot(centered_points, rotation_matrix.T)
                all_points[center_start_idx:center_end_idx] = rotated_points + shape_center
        
        # Get colors
        try:
            all_colors = apply_pca_and_store_colors(cluster_shapes, True).reshape(-1, 3)
        except:
            all_colors = np.tile([0.7, 0.5, 0.9], (len(all_points), 1))
        
        # Plot all shapes
        ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                c=all_colors, s=2, alpha=0.7)
        
        # Draw bounding box around the cluster center shape (now rotated)
        if center_position_in_grid is not None:
            # Get the points of the center shape in the grid (now includes rotation)
            center_start_idx = center_position_in_grid * K
            center_end_idx = (center_position_in_grid + 1) * K
            center_points = all_points[center_start_idx:center_end_idx]
            
            # Draw bounding box around center shape
            self.draw_shape_bounding_box(ax, center_points, color='red', linewidth=3, 
                                    label=f'Rotated Center (Cluster {cluster_id})', alpha=0.8)
            
            # Highlight the center shape points
            center_colors = all_colors[center_start_idx:center_end_idx]
            ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2],
                    c=center_colors, s=5, alpha=1.0, edgecolors='red', linewidths=0.5)
            
            # Add center label with rotation status
            center_bbox_center = np.mean(center_points, axis=0)
            
            # Check if center is rotated
            is_rotated = (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
                        cluster_id in self.processor.dino_data.cluster_center_rotations and
                        not np.allclose(self.processor.dino_data.cluster_center_rotations[cluster_id], np.eye(3)))
            
            # label_text = 'ROTATED\nCENTER' if is_rotated else 'CENTER'
            # label_color = 'red' if is_rotated else 'darkblue'
            
            # ax.text(center_bbox_center[0], center_bbox_center[1], center_bbox_center[2] + 0.3,
            #         label_text, fontsize=10, fontweight='bold', color=label_color, 
            #         ha='center', va='center',
            #         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add information about rotation behavior
        rotation_info = "Center rotated, others original" if (
            hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
            cluster_id in self.processor.dino_data.cluster_center_rotations and
            not np.allclose(self.processor.dino_data.cluster_center_rotations[cluster_id], np.eye(3))
        ) else "All shapes in original orientation"
        
        ax.text2D(0.02, 0.98, rotation_info, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9))




    def draw_shape_bounding_box(self, ax, points, color='red', linewidth=2, linestyle='-', 
                           alpha=1.0, label=None, padding=0.1):
        """Draw a 3D bounding box around a set of points"""
        
        # Calculate bounding box with padding
        min_coords = np.min(points, axis=0) - padding
        max_coords = np.max(points, axis=0) + padding
        
        # Define the 8 vertices of the bounding box
        vertices = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # 0: bottom-front-left
            [max_coords[0], min_coords[1], min_coords[2]],  # 1: bottom-front-right
            [max_coords[0], max_coords[1], min_coords[2]],  # 2: bottom-back-right
            [min_coords[0], max_coords[1], min_coords[2]],  # 3: bottom-back-left
            [min_coords[0], min_coords[1], max_coords[2]],  # 4: top-front-left
            [max_coords[0], min_coords[1], max_coords[2]],  # 5: top-front-right
            [max_coords[0], max_coords[1], max_coords[2]],  # 6: top-back-right
            [min_coords[0], max_coords[1], max_coords[2]]   # 7: top-back-left
        ])
        
        # Define the 12 edges of the bounding box
        edges = [
            # Bottom face
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face  
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        # Draw each edge
        for i, edge in enumerate(edges):
            start_vertex, end_vertex = vertices[edge[0]], vertices[edge[1]]
            
            # Add label only to the first edge to avoid duplicates
            edge_label = label if (i == 0 and label) else None
            
            ax.plot([start_vertex[0], end_vertex[0]], 
                    [start_vertex[1], end_vertex[1]], 
                    [start_vertex[2], end_vertex[2]], 
                    color=color, linewidth=linewidth, linestyle=linestyle, 
                    alpha=alpha, label=edge_label)


    # Alternative version with enhanced bounding box styles
    def draw_enhanced_shape_bounding_box(self, ax, points, color='red', style='solid', 
                                    padding=0.1, show_corners=False):
        """Draw an enhanced 3D bounding box with various styles"""
        
        min_coords = np.min(points, axis=0) - padding
        max_coords = np.max(points, axis=0) + padding
        
        vertices = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # 0
            [max_coords[0], min_coords[1], min_coords[2]],  # 1
            [max_coords[0], max_coords[1], min_coords[2]],  # 2
            [min_coords[0], max_coords[1], min_coords[2]],  # 3
            [min_coords[0], min_coords[1], max_coords[2]],  # 4
            [max_coords[0], min_coords[1], max_coords[2]],  # 5
            [max_coords[0], max_coords[1], max_coords[2]],  # 6
            [min_coords[0], max_coords[1], max_coords[2]]   # 7
        ])
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
        ]
        
        # Style configurations
        if style == 'solid':
            linewidth, linestyle, alpha = 3, '-', 0.8
        elif style == 'dashed':
            linewidth, linestyle, alpha = 2, '--', 0.7
        elif style == 'dotted':
            linewidth, linestyle, alpha = 2, ':', 0.6
        elif style == 'thick':
            linewidth, linestyle, alpha = 4, '-', 1.0
        else:
            linewidth, linestyle, alpha = 2, '-', 0.7
        
        # Draw edges
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                    color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        
        # Optionally draw corner markers
        if show_corners:
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    c=color, s=20, alpha=0.8, marker='o')



 
    def rotate_cluster_in_tab(self, cluster_id, axis):
        """Rotate ONLY the cluster center shape and update its specific tab"""
        # Get rotation angle for this cluster
        rotation_var = getattr(self, f'cluster_{cluster_id}_rotation_var', None)
        angle = rotation_var.get() if rotation_var else 15.0
        
        # Initialize rotation storage for cluster centers only
        if not hasattr(self.processor.dino_data, 'cluster_center_rotations'):
            self.processor.dino_data.cluster_center_rotations = {}
        
        if cluster_id not in self.processor.dino_data.cluster_center_rotations:
            self.processor.dino_data.cluster_center_rotations[cluster_id] = np.eye(3)
        
        # Create rotation matrix for this step
        angle_rad = np.radians(angle)
        if axis == 'x':
            step_rotation = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            step_rotation = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            step_rotation = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            self.log_cluster_status(cluster_id, f"Invalid rotation axis: {axis}")
            return
        
        # Accumulate rotation for CENTER ONLY
        current_rotation = self.processor.dino_data.cluster_center_rotations[cluster_id]
        self.processor.dino_data.cluster_center_rotations[cluster_id] = np.dot(step_rotation, current_rotation)
        
        # CRITICAL: Do NOT update cluster_rotations (this was causing all shapes to rotate)
        # Remove backward compatibility that was causing the issue
        
        # Update this cluster's tab visualization
        self.update_cluster_tab_visualization(cluster_id)
        
        # Log to cluster-specific status
        self.log_cluster_status(cluster_id, f"Center ONLY rotated by {angle}° around {axis.upper()}-axis")
        
        # Update main view to show rotated center (but not other shapes)
        if hasattr(self, 'viewer_update'):
            self.viewer_update('tsne')


    def update_cluster_center_with_rotation(self, cluster_id):
        """Apply current rotation to the cluster center shape"""
        if (cluster_id not in self.processor.dino_data.cluster_centers or
            not hasattr(self.processor.dino_data, 'cluster_center_rotations') or
            cluster_id not in self.processor.dino_data.cluster_center_rotations):
            return
        
        # Get original center shape
        center_data = self.processor.dino_data.cluster_centers[cluster_id]
        original_center_shape = center_data['center_shape'].copy()
        
        # Apply rotation to coordinates only
        rotation_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
        rotated_center_shape = original_center_shape.copy()
        rotated_center_shape[:, :3] = np.dot(original_center_shape[:, :3], rotation_matrix.T)
        
        # Store the rotated center shape
        center_data['rotated_center_shape'] = rotated_center_shape
        
        # Update the rotation matrix reference (for backward compatibility)
        if not hasattr(self.processor.dino_data, 'cluster_rotations'):
            self.processor.dino_data.cluster_rotations = {}
        self.processor.dino_data.cluster_rotations[cluster_id] = rotation_matrix


    # def apply_rotation_to_cluster_center(self, center_shape, cluster_id):
    #     """Apply rotation to cluster center shape using center-specific rotations"""
    #     if (not hasattr(self.processor.dino_data, 'cluster_center_rotations') or 
    #         cluster_id not in self.processor.dino_data.cluster_center_rotations):
    #         return center_shape
        
    #     rotation_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
        
    #     # If no rotation (identity matrix), return original
    #     if np.allclose(rotation_matrix, np.eye(3)):
    #         return center_shape
        
    #     # Apply rotation to the 3D coordinates (first 3 columns)
    #     rotated_shape = center_shape.copy()
    #     rotated_shape[:, :3] = np.dot(center_shape[:, :3], rotation_matrix.T)
        
    #     return rotated_shape
    


    def apply_rotation_to_cluster_center(self, center_shape, cluster_id):
        """Apply rotation to cluster center shape using center-specific rotations ONLY"""
        # ONLY check cluster_center_rotations - ignore old cluster_rotations
        if (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
            cluster_id in self.processor.dino_data.cluster_center_rotations):
            rotation_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
            
            # Apply rotation
            rotated_shape = center_shape.copy()
            rotated_shape[:, :3] = np.dot(center_shape[:, :3], rotation_matrix.T)
            return rotated_shape
        
        # NO FALLBACK to cluster_rotations - this was causing duplication
        return center_shape





    def reset_cluster_rotation_in_tab(self, cluster_id):
        """Reset rotation for ONLY the cluster center and update its tab"""
        # Reset center rotation
        if not hasattr(self.processor.dino_data, 'cluster_center_rotations'):
            self.processor.dino_data.cluster_center_rotations = {}
        
        self.processor.dino_data.cluster_center_rotations[cluster_id] = np.eye(3)
        
        # Update cluster center
        self.update_cluster_center_with_rotation(cluster_id)
        
        # Remove any stored rotated center shape
        if (cluster_id in self.processor.dino_data.cluster_centers and 
            'rotated_center_shape' in self.processor.dino_data.cluster_centers[cluster_id]):
            del self.processor.dino_data.cluster_centers[cluster_id]['rotated_center_shape']
        
        # Reset backward compatibility rotation matrix
        if hasattr(self.processor.dino_data, 'cluster_rotations'):
            self.processor.dino_data.cluster_rotations[cluster_id] = np.eye(3)
        
        self.update_cluster_tab_visualization(cluster_id)
        self.log_cluster_status(cluster_id, "Center rotation reset")
        
        # Update main view
        if hasattr(self, 'viewer_update'):
            self.viewer_update('tsne')


    def close_cluster_tab(self, cluster_id):
        """Close a specific cluster tab"""
        if cluster_id in self.cluster_tabs:
            tab = self.cluster_tabs[cluster_id]['tab']
            self.notebook.forget(tab)
            
            # Clean up
            del self.cluster_tabs[cluster_id]
            self.active_cluster_tabs.discard(cluster_id)
            
            # Clean up cluster-specific variables
            for attr_name in dir(self):
                if attr_name.startswith(f'cluster_{cluster_id}_'):
                    delattr(self, attr_name)
            
            self.log_status(f"Closed Cluster {cluster_id} tab")


    # def get_cluster_rotation_info(self, cluster_id):
    #     """Get rotation information for display"""
    #     if (hasattr(self.processor.dino_data, 'cluster_rotations') and 
    #         cluster_id in self.processor.dino_data.cluster_rotations):
    #         rot_matrix = self.processor.dino_data.cluster_rotations[cluster_id]
    #         if not np.allclose(rot_matrix, np.eye(3)):
    #             return "Rotations applied"
    #         else:
    #             return "No rotations"
    #     return "No rotations"


    def get_cluster_rotation_info(self, cluster_id):
        """Get rotation information for display"""
        if (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
            cluster_id in self.processor.dino_data.cluster_center_rotations):
            rot_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
            if not np.allclose(rot_matrix, np.eye(3)):
                return "Center rotated"
            else:
                return "Center not rotated"
        return "Center not rotated"


    def log_cluster_status(self, cluster_id, message):
        """Log message to cluster-specific status"""
        status_text = getattr(self, f'cluster_{cluster_id}_status_text', None)
        if status_text:
            status_text.insert(tk.END, f"{message}\n")
            status_text.see(tk.END)


    def set_cluster_tab_bounds(self, ax, cluster_id, view_mode):
        """Set appropriate bounds for cluster tab visualization"""
        # Get all plotted points to determine bounds
        collections = ax.collections
        if not collections:
            return
        
        all_points = []
        for collection in collections:
            if hasattr(collection, '_offsets3d'):
                x, y, z = collection._offsets3d
                points = np.column_stack([x, y, z])
                all_points.append(points)
        
        if all_points:
            combined_points = np.vstack(all_points)
            
            # Calculate bounds with padding
            padding = 0.2
            x_range = combined_points[:, 0].max() - combined_points[:, 0].min()
            y_range = combined_points[:, 1].max() - combined_points[:, 1].min()
            z_range = combined_points[:, 2].max() - combined_points[:, 2].min()
            max_range = max(x_range, y_range, z_range) if max(x_range, y_range, z_range) > 0 else 1.0
            
            x_center = (combined_points[:, 0].max() + combined_points[:, 0].min()) / 2
            y_center = (combined_points[:, 1].max() + combined_points[:, 1].min()) / 2
            z_center = (combined_points[:, 2].max() + combined_points[:, 2].min()) / 2
            
            half_range = max_range * (0.5 + padding)
            ax.set_xlim(x_center - half_range, x_center + half_range)
            ax.set_ylim(y_center - half_range, y_center + half_range)
            ax.set_zlim(z_center - half_range, z_center + half_range)



    def add_coordinate_system_to_ax(self, ax):
        """Add coordinate system to any axis using data coordinates"""
        # Get the current axis limits to scale the arrows appropriately
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        
        # Calculate arrow length as a percentage of the data range
        arrow_scale = 0.15  # 15% of the range
        x_length = (xlim[1] - xlim[0]) * arrow_scale
        y_length = (ylim[1] - ylim[0]) * arrow_scale
        z_length = (zlim[1] - zlim[0]) * arrow_scale
        
        # Position origin at a corner of the plot (adjust as needed)
        origin_x = xlim[0] - (xlim[1] - xlim[0]) * 0.1  # 10% from left
        origin_y = ylim[0] - (ylim[1] - ylim[0]) * 0.1  # 10% from bottom
        origin_z = zlim[0] - (zlim[1] - zlim[0]) * 0.1  # 10% from front
        
        axes_data = [
            ([origin_x, origin_x + x_length], [origin_y, origin_y], [origin_z, origin_z], 'red', 'X'),
            ([origin_x, origin_x], [origin_y, origin_y + y_length], [origin_z, origin_z], 'green', 'Y'),
            ([origin_x, origin_x], [origin_y, origin_y], [origin_z, origin_z + z_length], 'blue', 'Z')
        ]
        
        for x_coords, y_coords, z_coords, color, label in axes_data:
            # Remove transform=ax.transAxes to use data coordinates
            ax.plot(x_coords, y_coords, z_coords,
                    color=color, linewidth=6, alpha=0.9)  # Increased linewidth
            
            # Position labels at arrow tips
            end_x, end_y, end_z = x_coords[1], y_coords[1], z_coords[1]
            
            # Add small offset for label positioning
            offset = max(x_length, y_length, z_length) * 0.1
            label_x = end_x + (offset if label == 'X' else 0)
            label_y = end_y + (offset if label == 'Y' else 0)
            label_z = end_z + (offset if label == 'Z' else 0)
            
            ax.text(label_x, label_y, label_z, label,
                    color=color, fontsize=14, fontweight='bold')  # Increased fontsize

    # Additional helper methods for cluster tab management
    def apply_rotation_to_cluster_shapes(self, cluster_id):
        """Apply current rotation to all shapes in the cluster"""
        # This would implement the canonicalization for this specific cluster
        self.log_cluster_status(cluster_id, "Applied rotation to all shapes in cluster")


    # TODO: save the current cluster Canonicalization result
    def export_cluster(self, cluster_id):
        """Export the current cluster with canonicalization results"""
        if (self.processor.dino_data is None or 
            not hasattr(self.processor.dino_data, 'cluster_centers') or
            cluster_id not in self.processor.dino_data.cluster_centers):
            self.log_cluster_status(cluster_id, "Error: Cluster data not found")
            return

        # Get export directory
        # export_dir = filedialog.askdirectory(title=f"Select Export Directory for Cluster {cluster_id}")
        # if not export_dir:
        #     self.log_cluster_status(cluster_id, "Export cancelled")
        #     return

        export_dir=str(self.res_save_directory_path.get()) # self.res_save_directory_path 

        try:
            # Create cluster-specific directory ## TODO : use self.res_save_directory_path 
            cluster_export_dir = os.path.join(export_dir, f"cluster_{cluster_id}")
            os.makedirs(cluster_export_dir, exist_ok=True)
            
            # Get cluster data
            center_data = self.processor.dino_data.cluster_centers[cluster_id]
            cluster_labels = self.processor.dino_data.cluster_labels
            cluster_mask = cluster_labels == cluster_id
            cluster_shape_indices = np.where(cluster_mask)[0]
            
            self.log_cluster_status(cluster_id, f"Exporting {len(cluster_shape_indices)} shapes...")
            
            # === 1. Export Cluster Metadata ===
            cluster_metadata = {
                "cluster_id": int(cluster_id),
                "num_shapes": int(center_data['num_shapes']),
                "representative_index": int(center_data.get('representative_index', -1)),
                "shape_indices": cluster_shape_indices.tolist(),
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "canonicalization_applied": hasattr(self.processor.dino_data, 'cluster_shape_rotations') and 
                                        cluster_id in self.processor.dino_data.cluster_shape_rotations
            }
            
            with open(os.path.join(cluster_export_dir, "cluster_metadata.json"), "w") as f:
                json.dump(cluster_metadata, f, indent=2)
            
            # === 2. Export Rotation Matrices ===
            rotation_data = {
                "cluster_center_rotation": None,
                "shape_rotations": {},
                "rotation_matrices": {},
                "shape_name_to_index_mapping": {}
            }
            
            # Get shape names list
            shape_names_list = getattr(self.processor.dino_data, 'kpts_data_name_list', None)
            if shape_names_list is None:
                self.log_cluster_status(cluster_id, "Warning: No shape names found, using indices")
                shape_names_list = [f"shape_{i}" for i in range(len(self.processor.dino_data.kpts_data))]
            
            # Export cluster center rotation matrix
            if (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
                cluster_id in self.processor.dino_data.cluster_center_rotations):
                center_rotation_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
                rotation_data["cluster_center_rotation"] = {
                    "rotation_matrix": center_rotation_matrix.tolist(),
                    "is_identity": bool(np.allclose(center_rotation_matrix, np.eye(3)))
                }
            
            # Export individual shape rotation angles and compute full rotation matrices
            if (hasattr(self.processor.dino_data, 'cluster_shape_rotations') and 
                cluster_id in self.processor.dino_data.cluster_shape_rotations):
                
                shape_rotations = self.processor.dino_data.cluster_shape_rotations[cluster_id]
                
                for shape_idx in cluster_shape_indices:
                    # Get shape name
                    shape_name = shape_names_list[shape_idx] if shape_idx < len(shape_names_list) else f"shape_{shape_idx}"
                    
                    # Store name to index mapping
                    rotation_data["shape_name_to_index_mapping"][shape_name] = int(shape_idx)
                    
                    # Get rotation angle (canonicalization rotation)
                    canon_angle = shape_rotations.get(shape_idx, 0.0)
                    
                    # Compute canonicalization rotation matrix
                    canon_rotation_matrix = self.angle_to_rotation_matrix_z(canon_angle)
                    
                    # Get center rotation matrix if it exists
                    center_rotation_matrix = np.eye(3)
                    if (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
                        cluster_id in self.processor.dino_data.cluster_center_rotations):
                        center_rotation_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
                    
                    # Compute combined rotation: first canonicalization, then center rotation
                    combined_rotation_matrix = np.dot(center_rotation_matrix, canon_rotation_matrix)
                    
                    rotation_data["shape_rotations"][shape_name] = {
                        "shape_index": int(shape_idx),
                        "shape_name": shape_name,
                        "canonicalization_angle_degrees": float(canon_angle),
                        "canonicalization_rotation_matrix": canon_rotation_matrix.tolist(),
                        "cluster_center_rotation_matrix": center_rotation_matrix.tolist(),
                        "combined_rotation_matrix": combined_rotation_matrix.tolist(),
                        "is_canonical": bool(canon_angle != 0.0)
                    }
                    
                    rotation_data["rotation_matrices"][shape_name] = combined_rotation_matrix.tolist()
            else:
                # No canonicalization applied, but may have center rotation
                center_rotation_matrix = np.eye(3)
                if (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
                    cluster_id in self.processor.dino_data.cluster_center_rotations):
                    center_rotation_matrix = self.processor.dino_data.cluster_center_rotations[cluster_id]
                
                for shape_idx in cluster_shape_indices:
                    # Get shape name
                    shape_name = shape_names_list[shape_idx] if shape_idx < len(shape_names_list) else f"shape_{shape_idx}"
                    
                    # Store name to index mapping
                    rotation_data["shape_name_to_index_mapping"][shape_name] = int(shape_idx)
                    
                    rotation_data["shape_rotations"][shape_name] = {
                        "shape_index": int(shape_idx),
                        "shape_name": shape_name,
                        "canonicalization_angle_degrees": 0.0,
                        "canonicalization_rotation_matrix": np.eye(3).tolist(),
                        "cluster_center_rotation_matrix": center_rotation_matrix.tolist(),
                        "combined_rotation_matrix": center_rotation_matrix.tolist(),
                        "is_canonical": False
                    }
                    
                    rotation_data["rotation_matrices"][shape_name] = center_rotation_matrix.tolist()
            
            # Save rotation data
            with open(os.path.join(cluster_export_dir, "rotation_matrices.json"), "w") as f:
                json.dump(rotation_data, f, indent=2)
            
            # === 3. Export Point Clouds ===
            point_clouds_dir = os.path.join(cluster_export_dir, "point_clouds")
            os.makedirs(point_clouds_dir, exist_ok=True)
            
            # Export original point clouds (before any rotations)
            original_dir = os.path.join(point_clouds_dir, "original")
            os.makedirs(original_dir, exist_ok=True)
            
            # Export canonicalized point clouds (after rotations)
            canonicalized_dir = os.path.join(point_clouds_dir, "canonicalized")
            os.makedirs(canonicalized_dir, exist_ok=True)
            
            for i, shape_idx in enumerate(cluster_shape_indices):
                # Get shape name
                shape_names_list = getattr(self.processor.dino_data, 'kpts_data_name_list', None)
                if shape_names_list is None:
                    shape_name = f"shape_{shape_idx}"
                else:
                    shape_name = shape_names_list[shape_idx] if shape_idx < len(shape_names_list) else f"shape_{shape_idx}"
                
                # Get current (canonicalized) shape data
                current_shape = self.processor.dino_data.kpts_data[shape_idx]
                current_points = current_shape[:, :3]
                
                # Compute original points by reversing all rotations
                original_points = self.compute_original_points(shape_idx, cluster_id, current_points)
                
                # Save original point cloud
                original_pcd = o3d.geometry.PointCloud()
                original_pcd.points = o3d.utility.Vector3dVector(original_points)
                original_filename = os.path.join(original_dir, f"{shape_name}_original.ply")
                o3d.io.write_point_cloud(original_filename, original_pcd)
                
                # Save canonicalized point cloud
                canonical_pcd = o3d.geometry.PointCloud()
                canonical_pcd.points = o3d.utility.Vector3dVector(current_points)
                canonical_filename = os.path.join(canonicalized_dir, f"{shape_name}_canonicalized.ply")
                o3d.io.write_point_cloud(canonical_filename, canonical_pcd)
                
                # Save feature data as numpy arrays
                np.save(os.path.join(point_clouds_dir, f"{shape_name}_features.npy"), current_shape)
            
            # === 4. Export Cluster Center ===
            center_dir = os.path.join(cluster_export_dir, "cluster_center")
            os.makedirs(center_dir, exist_ok=True)
            
            # Export cluster center shape (with current center rotation applied)
            center_shape = self.apply_rotation_to_cluster_center(center_data['center_shape'], cluster_id)
            center_points = center_shape[:, :3]
            
            center_pcd = o3d.geometry.PointCloud()
            center_pcd.points = o3d.utility.Vector3dVector(center_points)
            center_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for center
            o3d.io.write_point_cloud(os.path.join(center_dir, "cluster_center.ply"), center_pcd)
            
            # Save center shape features
            np.save(os.path.join(center_dir, "cluster_center_features.npy"), center_shape)
            
            # === 5. Export Visualization Data ===
            viz_dir = os.path.join(cluster_export_dir, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Export t-SNE positions if available
            if hasattr(self.processor.dino_data, 'tsne_results') and self.processor.dino_data.tsne_results is not None:
                # Get shape names for the cluster
                shape_names_list = getattr(self.processor.dino_data, 'kpts_data_name_list', None)
                cluster_shape_names = []
                
                for shape_idx in cluster_shape_indices:
                    if shape_names_list is None:
                        shape_name = f"shape_{shape_idx}"
                    else:
                        shape_name = shape_names_list[shape_idx] if shape_idx < len(shape_names_list) else f"shape_{shape_idx}"
                    cluster_shape_names.append(shape_name)
                
                tsne_data = {
                    "tsne_positions": self.processor.dino_data.tsne_results[cluster_shape_indices].tolist(),
                    "shape_names": cluster_shape_names,
                    "shape_indices": cluster_shape_indices.tolist(),
                    "shape_name_to_index_mapping": {name: int(idx) for name, idx in zip(cluster_shape_names, cluster_shape_indices)}
                }
                with open(os.path.join(viz_dir, "tsne_positions.json"), "w") as f:
                    json.dump(tsne_data, f, indent=2)
            
            # === 6. Create Summary Report ===
            summary_report = self.create_cluster_export_summary(cluster_id, rotation_data, cluster_metadata)
            with open(os.path.join(cluster_export_dir, "export_summary.txt"), "w") as f:
                f.write(summary_report)
            
            # === 7. Create Visualization Images ===
            try:
                self.create_cluster_export_visualizations(cluster_id, cluster_export_dir)
            except Exception as e:
                self.log_cluster_status(cluster_id, f"Warning: Could not create visualizations: {str(e)}")
            
            self.log_cluster_status(cluster_id, f"Export completed successfully!")
            self.log_cluster_status(cluster_id, f"Exported to: {cluster_export_dir}")
            self.log_cluster_status(cluster_id, f"Files exported:")
            self.log_cluster_status(cluster_id, f"  - Metadata and rotation matrices")
            self.log_cluster_status(cluster_id, f"  - {len(cluster_shape_indices)} point clouds (original & canonicalized)")
            self.log_cluster_status(cluster_id, f"  - Cluster center shape")
            self.log_cluster_status(cluster_id, f"  - Visualization data")
            
        except Exception as e:
            self.log_cluster_status(cluster_id, f"Export failed: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export cluster {cluster_id}: {str(e)}")

    def angle_to_rotation_matrix_z(self, angle_degrees):
        """Convert angle in degrees to rotation matrix around Z-axis"""
        angle_rad = np.radians(angle_degrees)
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    def compute_original_points(self, shape_idx, cluster_id, current_points):
        """Compute original points by reversing all applied rotations"""
        original_points = current_points.copy()
        
        # Reverse canonicalization rotation
        if (hasattr(self.processor.dino_data, 'cluster_shape_rotations') and 
            cluster_id in self.processor.dino_data.cluster_shape_rotations and
            shape_idx in self.processor.dino_data.cluster_shape_rotations[cluster_id]):
            
            canon_angle = self.processor.dino_data.cluster_shape_rotations[cluster_id][shape_idx]
            if canon_angle != 0:
                reverse_canon_matrix = self.angle_to_rotation_matrix_z(-canon_angle)
                original_points = np.dot(original_points, reverse_canon_matrix.T)
        
        # Reverse center rotation
        if (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
            cluster_id in self.processor.dino_data.cluster_center_rotations):
            
            center_rotation = self.processor.dino_data.cluster_center_rotations[cluster_id]
            if not np.allclose(center_rotation, np.eye(3)):
                reverse_center_matrix = center_rotation.T  # Transpose is inverse for rotation matrices
                original_points = np.dot(original_points, reverse_center_matrix.T)
        
        return original_points

    def create_cluster_export_summary(self, cluster_id, rotation_data, cluster_metadata):
        """Create a human-readable summary of the export"""
        summary = f"""
    CLUSTER {cluster_id} EXPORT SUMMARY
    {'='*50}

    Cluster Information:
    - Cluster ID: {cluster_id}
    - Number of shapes: {cluster_metadata['num_shapes']}
    - Representative shape index: {cluster_metadata['representative_index']}
    - Export timestamp: {cluster_metadata['export_timestamp']}
    - Canonicalization applied: {'Yes' if cluster_metadata['canonicalization_applied'] else 'No'}

    Cluster Center Rotation:
    """
        
        if rotation_data["cluster_center_rotation"]:
            center_rot = rotation_data["cluster_center_rotation"]
            summary += f"- Applied: {'No' if center_rot['is_identity'] else 'Yes'}\n"
            if not center_rot['is_identity']:
                summary += f"- Rotation matrix applied to cluster center\n"
        else:
            summary += "- No rotation applied to cluster center\n"
        
        summary += f"\nShape Canonicalization Results:\n"
        summary += f"{'Shape Name':<20} {'Shape Index':<12} {'Canon Angle':<12} {'Status':<15}\n"
        summary += f"{'-'*60}\n"
        
        for shape_name, rotation_info in rotation_data["shape_rotations"].items():
            shape_idx = rotation_info["shape_index"]
            angle = rotation_info["canonicalization_angle_degrees"]
            status = "Canonicalized" if rotation_info["is_canonical"] else "Original"
            summary += f"{shape_name:<20} {shape_idx:<12} {angle:<12.1f} {status:<15}\n"
        
        summary += f"\nFiles Exported:\n"
        summary += f"- cluster_metadata.json: Cluster information and shape indices\n"
        summary += f"- rotation_matrices.json: All rotation matrices for shapes\n"
        summary += f"- point_clouds/original/: Original point clouds (before rotations)\n"
        summary += f"- point_clouds/canonicalized/: Canonicalized point clouds (after rotations)\n"
        summary += f"- point_clouds/*_features.npy: Feature vectors for each shape (named by shape)\n"
        summary += f"- cluster_center/: Cluster center shape and features\n"
        summary += f"- visualization/: t-SNE positions and visualization data (with shape names)\n"
        summary += f"- export_summary.txt: This summary file\n"
        
        summary += f"\nRotation Matrix Usage:\n"
        summary += f"- rotation_matrices.json contains matrices keyed by shape names\n"
        summary += f"- Use 'combined_rotation_matrix' for total transformation\n"
        summary += f"- 'shape_name_to_index_mapping' provides name-to-index lookup\n"
        
        return summary

    def create_cluster_export_visualizations(self, cluster_id, export_dir):
        """Create visualization images for the exported cluster"""
        import matplotlib.pyplot as plt
        
        viz_dir = os.path.join(export_dir, "visualization")
        
        # Create before/after comparison if canonicalization was applied
        if (hasattr(self.processor.dino_data, 'cluster_shape_rotations') and 
            cluster_id in self.processor.dino_data.cluster_shape_rotations):
            
            cluster_labels = self.processor.dino_data.cluster_labels
            cluster_mask = cluster_labels == cluster_id
            cluster_shape_indices = np.where(cluster_mask)[0]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot rotation angles
            angles = []
            indices = []
            for shape_idx in cluster_shape_indices:
                angle = self.processor.dino_data.cluster_shape_rotations[cluster_id].get(shape_idx, 0.0)
                angles.append(angle)
                indices.append(shape_idx)
            
            ax1.bar(range(len(indices)), angles)
            ax1.set_xlabel('Shape Index')
            ax1.set_ylabel('Canonicalization Angle (degrees)')
            ax1.set_title(f'Cluster {cluster_id} - Canonicalization Angles')
            ax1.set_xticks(range(len(indices)))
            ax1.set_xticklabels([str(idx) for idx in indices], rotation=45)
            
            # Plot shape distribution
            ax2.hist(angles, bins=8, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Rotation Angle (degrees)')
            ax2.set_ylabel('Number of Shapes')
            ax2.set_title(f'Cluster {cluster_id} - Angle Distribution')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "canonicalization_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        self.log_cluster_status(cluster_id, "Visualization images created")




    def rotate_selected_cluster(self, axis):
        """Rotate the selected cluster center shape and update the view"""
        if (self.processor.dino_data is None or 
            not hasattr(self.processor.dino_data, 'cluster_centers') or
            self.processor.dino_data.cluster_centers is None):
            messagebox.showwarning("Warning", "Please run clustering first")
            return
        
        try:
            selected_cluster = int(self.selected_cluster.get())
        except:
            messagebox.showwarning("Warning", "Invalid cluster selection")
            return
            
        if selected_cluster not in self.processor.dino_data.cluster_centers:
            messagebox.showwarning("Warning", f"Cluster {selected_cluster} not found")
            return
        
        angle = self.rotation_angle_var.get()
        
        # CRITICAL FIX: Use cluster_center_rotations instead of cluster_rotations
        if not hasattr(self.processor.dino_data, 'cluster_center_rotations'):
            self.processor.dino_data.cluster_center_rotations = {}
        
        if self.processor.dino_data.cluster_center_rotations is None:
            self.processor.dino_data.cluster_center_rotations = {}
        
        # Get current rotation matrix or initialize to identity
        if selected_cluster not in self.processor.dino_data.cluster_center_rotations:
            self.processor.dino_data.cluster_center_rotations[selected_cluster] = np.eye(3)
        
        # Create rotation matrix for this step
        angle_rad = np.radians(angle)
        if axis == 'x':
            step_rotation = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            step_rotation = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            step_rotation = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            return
        
        # Accumulate rotation using cluster_center_rotations
        current_rotation = self.processor.dino_data.cluster_center_rotations[selected_cluster]
        self.processor.dino_data.cluster_center_rotations[selected_cluster] = np.dot(step_rotation, current_rotation)
        
        self.log_status(f"Rotated cluster {selected_cluster} center by {angle}° around {axis.upper()}-axis")
        
        # CRITICAL FIX: Complete refresh instead of partial update
        self.viewer_update('tsne')


    def update_cluster_centers_display(self):
        """DEPRECATED - Use viewer_update instead for complete refresh"""
        # This function causes duplication issues - use complete refresh instead
        self.log_status("Warning: update_cluster_centers_display is deprecated. Using complete refresh.")
        self.viewer_update('tsne')



    

    def draw_cluster_center_points_only(self):
        """Draw only the cluster center points with proper rotation handling"""
        cluster_centers_data = self.processor.dino_data.cluster_centers
        unique_labels = list(cluster_centers_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        try:
            selected_cluster = int(self.selected_cluster.get())
        except:
            selected_cluster = -1
        
        for i, (label, center_data) in enumerate(cluster_centers_data.items()):
            center_shape = center_data['center_shape']
            tsne_position = center_data['tsne_position']
            
            # CRITICAL FIX: Use apply_rotation_to_cluster_center for consistency
            rotated_center_shape = self.apply_rotation_to_cluster_center(center_shape, label)
            center_points = rotated_center_shape[:, :3]
            
            # Apply t-SNE positioning
            positioned_center_points = center_points + tsne_position
            
            # Use all points for better visualization
            display_points = positioned_center_points
            
            color = colors[i]
            
            # Check if this cluster has been rotated for visual distinction
            is_rotated = (hasattr(self.processor.dino_data, 'cluster_center_rotations') and 
                        label in self.processor.dino_data.cluster_center_rotations and
                        not np.allclose(self.processor.dino_data.cluster_center_rotations[label], np.eye(3)))
            
            # Visual styling based on selection and rotation status
            if label == selected_cluster:
                edge_color = 'red' if is_rotated else 'black'
                point_size = 10 if is_rotated else 8
                alpha = 1.0
            else:
                edge_color = 'darkred' if is_rotated else None
                point_size = 7 if is_rotated else 5
                alpha = 0.9 if is_rotated else 0.7
            
            self.ax_3d.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                            c=[color], s=point_size, alpha=alpha,
                            edgecolors=edge_color, linewidths=1.0 if edge_color else 0)






    def update_tsne_view_with_rotation(self):
        """Update the t-SNE view showing all shapes with current rotations applied"""
        if self.processor.dino_data is None:
            return
        
        self.ax_3d.clear()
        
        kpts_data = self.processor.dino_data.kpts_data
        cluster_labels = self.processor.dino_data.cluster_labels
        
        # Use stored t-SNE results and cluster data
        if (not hasattr(self.processor.dino_data, 'tsne_results') or 
            self.processor.dino_data.tsne_results is None):
            # If no stored results, recompute (fallback)
            num_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
            self.show_re_clustering_tsne_view(num_clusters)
            return
        
        n_shapes = len(kpts_data)
        tsne_results = self.processor.dino_data.tsne_results
        
        # Create t-SNE centers
        tsne_center = np.zeros_like(kpts_data[:n_shapes, 0, :3])
        tsne_center[:, :2] = tsne_results[:n_shapes]
        
        # Plot all original shapes
        all_colors = apply_pca_and_store_colors(kpts_data, True)[:n_shapes].reshape(-1, 3)
        all_points = kpts_data[:n_shapes, :, :3]
        K = all_points.shape[1]
        
        all_points_ = kpts_data[:n_shapes, :, :3].reshape(-1, 3) + tsne_center[:, None, :].repeat(K, axis=1).reshape(-1, 3)
        all_colors_ = all_colors
        
        # Plot all shapes (background)
        self.ax_3d.scatter(all_points_[:, 0], all_points_[:, 1], all_points_[:, 2],c=all_colors_, s=1, alpha=0.3)  # Reduced alpha for background
        
        # Plot cluster centers with rotations applied
        if hasattr(self.processor.dino_data, 'cluster_centers') and self.processor.dino_data.cluster_centers:
            self.draw_cluster_centers_with_rotations()
        
        self.canvas_3d.mpl_connect('scroll_event', self.on_scroll)
        self.ax_3d.set_axis_off()
        self.ax_3d.grid(False)
        self.ax_3d.set_title('All Loaded Shapes (t-SNE View) - Rotations Applied')
        
        # Set equal aspect ratio
        x_range = all_points_[:, 0].max() - all_points_[:, 0].min()
        y_range = all_points_[:, 1].max() - all_points_[:, 1].min()
        z_range = all_points_[:, 2].max() - all_points_[:, 2].min()
        max_range = max(x_range, y_range, z_range)
        
        x_center = (all_points_[:, 0].max() + all_points_[:, 0].min()) / 2
        y_center = (all_points_[:, 1].max() + all_points_[:, 1].min()) / 2
        z_center = (all_points_[:, 2].max() + all_points_[:, 2].min()) / 2
        
        self.ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
        self.ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
        self.ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        self.fig_3d.tight_layout(pad=0)
        self.canvas_3d.draw_idle()



    def reset_cluster_rotation(self):
        """Reset rotation for the selected cluster and visualize immediately"""
        if (self.processor.dino_data is None or 
            not hasattr(self.processor.dino_data, 'cluster_centers')):
            return
        
        try:
            selected_cluster = int(self.selected_cluster.get())
        except:
            return
        
        # CRITICAL FIX: Use cluster_center_rotations instead of cluster_rotations
        if not hasattr(self.processor.dino_data, 'cluster_center_rotations'):
            self.processor.dino_data.cluster_center_rotations = {}
        
        # Reset the center rotation
        self.processor.dino_data.cluster_center_rotations[selected_cluster] = np.eye(3)
        
        self.log_status(f"Reset rotation for cluster {selected_cluster} center")
        
        # CRITICAL FIX: Complete refresh instead of partial update
        self.viewer_update('tsne')


    def show_tsne_plot(self):
        """Show t-SNE visualization of clusters"""
        if (self.processor.dino_data is None or 
            self.processor.dino_data.tsne_results is None):
            return
        
        # Switch to statistics tab
        self.notebook.select(1)
        
        # Clear previous plot
        self.ax_stats.clear()
        
        tsne_results = self.processor.dino_data.tsne_results
        cluster_labels = self.processor.dino_data.cluster_labels
        
        # Create scatter plot colored by clusters
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            if label == -1:  # Noise points
                self.ax_stats.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                                     c='gray', alpha=0.6, s=20, label='Noise')
            else:
                self.ax_stats.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                                     c=[colors[i]], alpha=0.7, s=20, label=f'Cluster {label}')
        
        self.ax_stats.set_xlabel('t-SNE Dimension 1')
        self.ax_stats.set_ylabel('t-SNE Dimension 2')
        self.ax_stats.set_title('t-SNE Visualization of Shape Clusters')
        self.ax_stats.legend()
        self.ax_stats.grid(True, alpha=0.3)
        
        self.canvas_stats.draw()
    
    def show_statistics(self):
        """Show clustering statistics in the statistics tab"""
        if (self.processor.dino_data is None or 
            self.processor.dino_data.cluster_labels is None):
            return
        
        labels = self.processor.dino_data.cluster_labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Clear previous plot
        self.ax_stats.clear()
        
        # Create bar plot
        bars = self.ax_stats.bar(range(len(unique_labels)), counts)
        
        # Color bars according to cluster colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        self.ax_stats.set_xlabel('Cluster ID')
        self.ax_stats.set_ylabel('Number of Shapes')
        self.ax_stats.set_title('Shapes per Cluster')
        self.ax_stats.set_xticks(range(len(unique_labels)))
        self.ax_stats.set_xticklabels([f'{label}' for label in unique_labels])
        
        # Add statistics text
        stats_text = f"Total Shapes: {len(labels)}\n"
        stats_text += f"Number of Clusters: {len(unique_labels[unique_labels != -1])}\n"
        if -1 in unique_labels:
            noise_count = counts[unique_labels == -1][0]
            stats_text += f"Noise Shapes: {noise_count}\n"
        
        self.ax_stats.text(0.02, 0.98, stats_text, transform=self.ax_stats.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.canvas_stats.draw()
    
    def reset_3d_view(self):
        """Reset the 3D view to default"""
        if self.processor.dino_data is not None:
            if self.processor.dino_data.cluster_centers is not None:
                self.show_cluster_centers()
            else:
                self.show_all_shapes()
        else:
            self.show_instructions()
    
    def save_result(self):
        """Save the current clustering result"""
        if self.processor.dino_data is None:
            messagebox.showwarning("Warning", "No data to save")
            return
        
        directory_path = filedialog.askdirectory(title="Select Save Directory")
        
        if directory_path:
            try:
                # Save cluster labels
                if self.processor.dino_data.cluster_labels is not None:
                    np.save(os.path.join(directory_path, "cluster_labels.npy"), 
                           self.processor.dino_data.cluster_labels)
                
                # Save t-SNE results
                if self.processor.dino_data.tsne_results is not None:
                    np.save(os.path.join(directory_path, "tsne_results.npy"), 
                           self.processor.dino_data.tsne_results)
                
                # Save cluster centers
                if self.processor.dino_data.cluster_centers is not None:
                    np.save(os.path.join(directory_path, "cluster_centers.npy"), 
                           self.processor.dino_data.cluster_centers)
                
                # Save rotation matrices
                if self.processor.dino_data.cluster_rotations:
                    with open(os.path.join(directory_path, "cluster_rotations.json"), "w") as f:
                        # Convert numpy arrays to lists for JSON serialization
                        rotations_serializable = {
                            str(k): v.tolist() for k, v in self.processor.dino_data.cluster_rotations.items()
                        }
                        json.dump(rotations_serializable, f, indent=2)
                
                self.log_status(f"Results saved to: {directory_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def further_processing(self):
        """Placeholder for further processing operations"""
        if (self.processor.dino_data is None or 
            self.processor.dino_data.cluster_labels is None):
            messagebox.showwarning("Warning", "Please run clustering first")
            return
        
        self.log_status("Running further processing...")
        
        # Example: Process each cluster separately
        unique_labels = np.unique(self.processor.dino_data.cluster_labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            
            cluster_mask = self.processor.dino_data.cluster_labels == label
            cluster_shapes = self.processor.dino_data.kpts_data[cluster_mask]
            
            # Add your specific processing for each cluster here
            self.log_status(f"Processing cluster {label}: {len(cluster_shapes)} shapes")
        
        self.log_status("Further processing complete")
    
    def export_results(self):
        """Export results in various formats"""
        if self.processor.dino_data is None:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        try:
            # Export cluster information as JSON
            if self.processor.dino_data.cluster_labels is not None:
                unique_labels, counts = np.unique(self.processor.dino_data.cluster_labels, return_counts=True)
                cluster_info = {
                    "total_shapes": len(self.processor.dino_data.cluster_labels),
                    "num_clusters": len(unique_labels[unique_labels != -1]),
                    "cluster_sizes": {int(label): int(count) for label, count in zip(unique_labels, counts)}
                }
                
                with open(os.path.join(export_dir, "cluster_info.json"), "w") as f:
                    json.dump(cluster_info, f, indent=2)
            
            # Export individual cluster center shapes as PLY files
            if self.processor.dino_data.cluster_centers is not None:
                cluster_labels = self.processor.dino_data.cluster_labels
                unique_labels = np.unique(cluster_labels)
                valid_labels = unique_labels[unique_labels != -1]
                
                for i, label in enumerate(valid_labels):
                    center_shape = self.processor.dino_data.cluster_centers[i]
                    points = center_shape[:, :3]
                    
                    # Apply rotations if any
                    if (self.processor.dino_data.cluster_rotations and 
                        label in self.processor.dino_data.cluster_rotations):
                        rotation_matrix = self.processor.dino_data.cluster_rotations[label]
                        points = np.dot(points, rotation_matrix.T)
                    
                    # Create Open3D point cloud and save
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.paint_uniform_color([0.5, 0.5, 0.5])
                    
                    filename = os.path.join(export_dir, f"cluster_{label}_center.ply")
                    o3d.io.write_point_cloud(filename, pcd)
            
            self.log_status(f"Results exported to: {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

# Example usage and integration points for your existing pipeline
class YourPipelineIntegration:
    """
    This class shows how to integrate your existing pipeline functions
    Replace the methods below with your actual implementations
    """
    
    @staticmethod
    def read_shape_multi_views(shape_path: str) -> List[np.ndarray]:
        """Your function to read shape multi-views"""
        # Replace with your actual implementation
        pass
    
    @staticmethod
    def do_backprojection(views: List[np.ndarray]) -> np.ndarray:
        """Your function to do backprojection"""
        # Replace with your actual implementation
        pass
    
    @staticmethod
    def dino_feature_lifting(points: np.ndarray, views: List[np.ndarray]) -> np.ndarray:
        """Your function to do DINO feature lifting"""
        # Replace with your actual implementation
        pass


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nReceived Ctrl+C, shutting down...')
    sys.exit(0)

def main():
    """Main function to run the application"""
    signal.signal(signal.SIGINT, signal_handler)
    try:
        # Create and run the GUI application
        app = ShapeCanonicalizeGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":


    ## env: conda activate diffTheta3D && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    # export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    main()