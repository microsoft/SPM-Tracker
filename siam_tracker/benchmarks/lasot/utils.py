import os
import numpy as np


def load_avaiable_dir_list(root_dir='data/benchmarks/lasot'):

    category_dir_list = [os.path.join(root_dir, fn) for fn in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, fn))]
    # print(category_dir_list)
    avaiable_dir_list = []
    for category_dir in category_dir_list:
        subcat_count = 0
        subcat_dir_list = [os.path.join(category_dir, fn) for fn in os.listdir(category_dir)
            if os.path.isdir(os.path.join(category_dir, fn))]
        for subcat_dir in subcat_dir_list:
            if os.path.exists(os.path.join(subcat_dir, 'groundtruth.txt')) and os.path.isdir(os.path.join(subcat_dir, 'img')):
                avaiable_dir_list.append(subcat_dir)
                subcat_count += 1
        
        print("[LaSOT] Loading '{}', {} sequences".format(os.path.basename(category_dir), subcat_count))
    
    return avaiable_dir_list


def load_video_information(seq_dir):

    gt_path = os.path.join(seq_dir, 'groundtruth.txt')
    img_dir = os.path.join(seq_dir, 'img')
    
    img_list = [fn for fn in os.listdir(img_dir) 
        if fn.split('.')[-1].lower() in ('jpg', 'png', 'jpeg', 'bmp')]
    
    img_inds_list = [int(fn.split('.')[0]) for fn in img_list]
    # img_inds_list_np = np.array(img_inds_list, dtype=int)
    sort_inds = np.argsort(img_inds_list)
    img_list = [img_list[i] for i in sort_inds]
    # print(img_list)
    full_img_list = [os.path.join(seq_dir, 'img', fn) for fn in img_list]
    with open(gt_path, 'r') as f:
        lines = f.read().splitlines() 
    lines = [line for line in lines if line.strip() != '']
    
    gt_rects = np.zeros((len(lines), 4), np.float32)
    for i in range(len(lines)):
        sp = lines[i].split(',')
        x1 = float(sp[0])
        y1 = float(sp[1])
        x2 = float(sp[2]) + x1
        y2 = float(sp[3]) + y1
        gt_rects[i, :] = [x1, y1, x2, y2]

    return full_img_list, gt_rects


def write_result(output_path, tracking_results):

    xywh = [[t[0], t[1], t[2] - t[0], t[3] - t[1]] for t in tracking_results]
    with open(output_path, 'w') as f:
        for t in xywh:
            f.write('{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(t[0], t[1], t[2], t[3]))
    
        
