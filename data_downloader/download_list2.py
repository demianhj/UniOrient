
import os
import json
import pandas as pd
import json, os, itertools
import subprocess

metadata_objaverse_github_path='./save_metadata_obja_github.csv'
metadata_objaverse_sketchfab_path='./ObjaverseXL_sketchfab.csv'
metas= [metadata_objaverse_github_path, metadata_objaverse_sketchfab_path]



gobjaverse_280k_Furnitures = './gobjaverse_280k_Furnitures.json'
fbx_map_json_path='./fbx_map.json'
obj_map_json_path='./obj_map.json'
sketchfab_map_json_path='./sketchfab_map.json'
glb_map_json_path='./glb_map.json'




def load_keys(path):
    """Return a list of cleaned keys from one JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # the files are lists of one-key dictionaries
    #  – flatten to (key, value) pairs
    if isinstance(data, list):
        items = itertools.chain.from_iterable(d.items() for d in data)
    else:  # just in case the structure is {key: url, …}
        items = data.items()

    cleaned = []
    for k, val in items:
        # drop ".tar.gz" (or any extension after the first dot)
        base = os.path.splitext(os.path.splitext(k)[0])[0]
        # base_val= os.path.splitext(os.path.splitext(val)[0])[0]
        # # print("show base:",base)

    
        
        cleaned.append(base)

    
    return cleaned



all_paths = [
    fbx_map_json_path,
    obj_map_json_path,
    sketchfab_map_json_path,
    glb_map_json_path,
]

def load_all_maps(paths):
    merged_map = {}
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    merged_map.update(item)  # Assuming format: [{key: url}, ...]
        else:
            print(f"⚠️ Missing file: {path}")
    return merged_map

# collect and de-duplicate
# all_keys = sorted(set(itertools.chain.from_iterable(load_keys(p) for p in all_paths)))



def download_url(url, save_dir):
    subfolder = url.split('/')[-2]
    output_dir = os.path.join(save_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    command = ['wget', '-P', output_dir, url]
    subprocess.run(command, check=True)

if __name__=='__main__':

    output_path = "./data_downloader/core_data/474_category_id_to_urls.json"

    num_categories= '/home/lei/Documents/Dataset/Canonicalization/lvis50/core_data/categories_more_than_20.json'

    # with open(num_categories, 'r', encoding='utf-8') as f:
    #         num_categories_values = json.load(f) 
    

    # sorted_by_count = sorted(
    # num_categories_values.items(),
    # key=lambda item: item[1],
    # reverse=True
    # )

    # # 3) If you just want the category names in that order:
    # sorted_categories = [cat for cat, count in sorted_by_count]

    print("Categories sorted by count (desc):")

   

    download_root= './data/G-objaverse'
    folders = [folder for folder in os.listdir(download_root)]
    print("see folders:", folders)
    

    # Load category-wise ID → URL mapping
    with open(output_path, 'r', encoding='utf-8') as f:
        category_id_to_url = json.load(f)


    counter=0
    # Download each file into category-named folder
    for category, id_url_map in category_id_to_url.items():

        # armchair
        if category !='trash_can': # armor # statue_(sculpture) # flower_arrangement # barrel # mushroom
            # vase #armchair # air_conditioner
            # new# doll # figurine # airplane
            
            continue

        category_dir = os.path.join(download_root, category)
        print(f"📁 Downloading models for category: {category}")
       
        for shape_id, url in id_url_map.items():
            # todo: use shape_id in the following:
            download_path = f"https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_alignment/{shape_id}.tar.gz"

            # download_path = f"https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/gobjaverse_alignment/1090/5456862.tar.gz"

            # 

            shape_id= os.path.basename(download_path)
            subfolder = download_path.split('/')[-2]
            # print("see existing res",subfolder)
            res =  os.path.join(category_dir, subfolder,shape_id)
      
            if os.path.exists(res):
                print("see existing res",res)
                counter+=1
                continue


            try:
                print("show url to be downloaded:", download_path)
                download_url(download_path, category_dir)
            except Exception as e:
                print(f"❌ Failed to download {download_path}: {e}")

        print(f"📁 Downloading models for category: {counter}")

    # conda activate data_anno && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    # python download_list.py
    # export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH




    # conda activate data_anno 
    # python data_downloader/download_list2.py
