import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import scipy 
from scipy.stats import sem, tstd 
import os 
import matplotlib as mpl 
import seaborn as sns 
import pandas as pd 
from matplotlib.patches import Patch 
from matplotlib.lines import Line2D 

## 8 keypoints for evaluation in total. 
keypoint_names_for_eval = [
    "Nose", 
    "Tail", 
    "LPaw", 
    "RPaw", 
    "LFoot",
    "RFoot", 
    "LEar",
    "REar"
]

mapper = [2, 7, 8,12, 16, 19, 0, 1]

with open('data/markerless_mouse_1_nerf/label_ids_mid.pkl', 'rb') as f: 
    frames_to_eval = pickle.load(f) 

def load_gt(): 
    with open("data/markerless_mouse_1_nerf/add_labels_3d_8keypoints.pkl", 'rb') as f: 
        all_gt = pickle.load(f)
    return np.asarray(all_gt) 

def load_dannce_temp_predict(): 
    rawdata = scipy.io.loadmat("/home/animal/projects/dannce-pytorch/demo/markerless_mouse_1/DANNCE/predict_results/save_data_AVG0.mat")
    sampleIDs = rawdata["sampleID"].astype(np.int64) 
    pred = rawdata["pred"] # N, 3, 22 
    data = rawdata["data"] # N, 3, 22
    p_max = rawdata["p_max"] # N, 22 
    pred_eval = pred[frames_to_eval ]
    pred_eval_11 = pred_eval[:,:,mapper]
    pred_out = np.transpose(pred_eval_11, (0,2,1))
    return pred_out

def load_dannce_raw_predict(): 
    rawdata = scipy.io.loadmat("/home/animal/projects/dannce/demo/markerless_mouse_1/DANNCE/predict_results_bk/save_data_AVG0.mat")
    sampleIDs = rawdata["sampleID"].astype(np.int64) 
    pred = rawdata["pred"] # N, 3, 22 
    data = rawdata["data"] # N, 3, 22
    p_max = rawdata["p_max"] # N, 22 
    pred_eval = pred[frames_to_eval ]
    pred_eval_11 = pred_eval[:,:,mapper]
    pred_out = np.transpose(pred_eval_11, (0,2,1))
    return pred_out

def load_fitting_result(filename): 
    with open("mouse_fitting_result/eval_tmp/" + filename, 'rb') as f: 
        data = pickle.load(f) 

    return data 

def standard_deviation(data):
    mean = data.mean()  
    sd = ( (data - mean) ** 2).mean()
    sd = np.sqrt(sd) 
    return sd 
  
def evaluate(gt, pred): 
    all_errors = np.zeros([50,8]) - 1 
    for fid in range(50): 
        for kid in range(8): 
            gt_point = gt[fid, kid]
            pred_point = pred[fid, kid] 
            if np.linalg.norm(gt_point) == 0: 
                continue 
            e = np.linalg.norm(gt_point-pred_point) 
            all_errors[fid, kid] = e 

    valid_errors = all_errors[all_errors > 0]
    mean_err = valid_errors.mean()  
    sd = standard_deviation(valid_errors)
    print("  avg: ", mean_err)
    print("  sd  : ", sd)
    output_dict = {} 
    output_dict["mean"] = mean_err
    output_dict["sd"] = sd 
    for k in range(8): 
        joint_e = all_errors[:,k]
        error = joint_e[joint_e > 0].mean() 
        print("  ", keypoint_names_for_eval[k], error)
        pck = ( (joint_e < 3) & (joint_e >= 0) ).sum() / 50
        print("      ", pck * 100, " %")
        sd = standard_deviation(joint_e[joint_e > 0]) 
        print("      ", sd)
        output_dict["mean_{}".format(k)] = error 
        output_dict["sd_{}".format(k)] = sd 
    output_dict["all_errors"] = all_errors 
    return output_dict

def compare_all(): 
    all_gt = load_gt() 
    dannce_temp = load_dannce_temp_predict() 
    fit_6 = load_fitting_result("fit_6view.pkl") 

    output_dict = {} 
    print("DANNCE_temp: ")
    data_DANNCE_temp = evaluate(all_gt, dannce_temp)
    output_dict["dannce"] = data_DANNCE_temp

    print("MAMMAL: ")
    data_MAMMAL = evaluate(all_gt, fit_6)
    output_dict["MAMMAL"] = data_MAMMAL 

    os.makedirs("tmp_eval", exist_ok=True)
    with open("tmp_eval/result.pkl", 'wb') as f: 
        pickle.dump(output_dict, f) 

def build_data_frame(): 
    with open("tmp_eval/result.pkl", 'rb') as f: 
        data_dict = pickle.load(f) 
    ## only evaluate tracked points, DO NOT compare untracked points. This may overestimate the performance of SLEAP-tri. 
    errors = [] 
    method = [] 
    jointname = [] 
    rawframeids = [] 
    with open("data/markerless_mouse_1_nerf/label_ids_mid.pkl", 'rb') as f: 
        label_ids = pickle.load(f) 
    raw_method_name = ["dannce", "MAMMAL"]
    used_name = ["DANNCE-T", "MAMMAL"]
    for method_id in range(2): 
        method_key = raw_method_name[method_id]
        method_name = used_name[method_id]
        data = data_dict[method_key]["all_errors"]

        for frameid in range(50): 
            for keypoint_id in range(8): 
                if data[frameid, keypoint_id] <= 0: 
                    continue 
                errors.append(data[frameid, keypoint_id]) 
                method.append(method_name)
                jointname.append(keypoint_names_for_eval[keypoint_id])
                rawframeids.append(label_ids[frameid])
    data_frame = { 
        "error": errors, 
        "method": method, 
        "jointname": jointname,
        "frameid": rawframeids
    }
    return pd.DataFrame(data = data_frame)
    
def plot_figure(): 
    os.makedirs("figs", exist_ok=True)
    mpl.rc('font', family='arial') 

    fig = plt.figure(figsize=(4.5,1.3)) 
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    color_maps = np.loadtxt("colormaps/anliang_paper.txt") / 255
    colors = {
        "DANNCE-T": color_maps[0],
        "MAMMAL": color_maps[1]
    }
    used_name = ["DANNCE-T", "MAMMAL"]

    data_frame = build_data_frame() 
    ### generating xlsx 
    data_frame.to_excel("figs/mouse_data.xlsx", sheet_name="data")
    ### end. 
    ax = sns.boxplot(x="jointname", y="error", hue="method", data=data_frame, palette=colors,
        linewidth=0.5, sym="", showmeans=True,
        
        meanprops = {'marker':'s','markerfacecolor':'black','markeredgecolor':'black', 'linewidth':0, 'markersize':1.5}, 
        capprops={"linewidth":0.5, "color": 'k'},
        whiskerprops={"linewidth":0.5, "color": "k"},
        )
    # ax.plot([-0.67,18.23], [0.07,0.07], linestyle='--',linewidth=0.5, color = 'g')
    plt.xticks(rotation=0, ha='center', fontsize=7)
    legend_elements = [
        Patch(facecolor=color_maps[0], edgecolor='black', label=used_name[0], linewidth=0.5), 
        Patch(facecolor=color_maps[1], edgecolor='black',label=used_name[1], linewidth=0.5),
        # Patch(facecolor=color_maps[2], edgecolor='black',label=used_name[2], linewidth=0.5),
        # Patch(facecolor=color_maps[3], edgecolor='black',label=used_name[3], linewidth=0.5)
    ]
    ax = fig.get_axes()[0]
    for line in ["bottom", "left", "right", "top"]: 
        ax.spines[line].set_linewidth(0.5)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)

    plt.legend(handles=legend_elements, fontsize=7, ncol=1, loc='best', frameon=False)
    plt.xlabel("", fontsize=7)
    plt.ylabel("Error (mm)", fontsize=7)
    plt.ylim(0,20)
    plt.yticks([0,5,10,15,20], labels=[0,5,10,15,20], fontsize=7)
    plt.savefig("figs/Fig_compare_dannce2.png", dpi=1000, bbox_inches='tight', pad_inches=0.01)
    plt.savefig("figs/Fig_compare_dannce2.svg", dpi=1000, bbox_inches='tight', pad_inches=0.01) # uncomment this to write vector image.

if __name__ == "__main__": 
    compare_all()

    plot_figure()
