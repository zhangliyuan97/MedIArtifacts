import os
import sys
sys.path.append(os.path.abspath('your cls_task absolute path'))
import yaml
import shutil
import random
import argparse
from tqdm import tqdm

import torch

from datasets import create_dataset
from models import get_resnet_model
from utils.metrics import get_auc, get_accuracy, get_sensitivity, get_specifity
from utils.plot_roc_curve import plot_roc_curve


@torch.no_grad()
def test(cfg, model_path, figure_dir):
    """ Test function for evaluating hematoma expansions test dataset
    
    :param cfg: Configuration file for classification trainers, we used it to load Network architecture.
    :model_path: pretrained classifier for torch.load().
    :figure_dir: we will plot ROC/PR curve and save these curves in this direcory.
    """

    datasets = create_dataset(cfg)
    test_loader = datasets.test_loader
    
    cls_net = get_resnet_model(cfg["model"]["arch"], in_channels=cfg["in_channels"], n_classes=cfg["n_classes"])
    cls_net.load_state_dict(torch.load(model_path))
    cls_net = cls_net.cuda()
    cls_net.eval()
    
    epoch_outputs = []
    epoch_labels = []
    for test_batch in tqdm(test_loader):
        
        inputs = test_batch['inputs'].cuda()
        cls_labels = test_batch['cls_labels']
        
        outputs = cls_net(inputs)
        
        epoch_outputs.append(torch.sigmoid(outputs).detach().data.cpu())
        epoch_labels.append(cls_labels)
    
    epoch_outputs = torch.cat(epoch_outputs).numpy()
    epoch_labels = torch.cat(epoch_labels).numpy()
    
    test_auc = get_auc(epoch_outputs, epoch_labels)
    test_accuracy = get_accuracy(epoch_outputs, epoch_labels)
    test_sensitivity = get_sensitivity(epoch_outputs, epoch_labels)
    test_specificity = get_specifity(epoch_outputs, epoch_labels)
    
    print("AUC: {:.3f}, Acc: {:.3f}, Sen: {:.3f}, Spe: {:.3f}".format(test_auc, test_accuracy, test_sensitivity, test_specificity))
    
    # plot ROC curve and save it to figure dir
    roc_curve_file = os.path.join(figure_dir, "roc.png")
    plot_roc_curve(epoch_outputs, epoch_labels, roc_curve_file)
    
    # =========================================================================
    # plot other evaluation figures, such as confusion matrix, Grad-CAM, etc...
    # =========================================================================

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hematoma expansion prediction setting options")
    parser.add_argument(
        "--config", nargs="?", type=str, 
        default="runs/hematoma/res10_in/hematoma.yml", 
        help="Configuration for hematoma expansion testing, used by previous trainers"
    )
    parser.add_argument(
        "--model_path", nargs="?", type=str, 
        default="runs/hematoma/res10_in/best_auc_model/best_model.pt", 
        help="Configuration for hematoma expansion testing, used by previous trainers"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    # cfg['local_rank'] = args.local_rank
    # torch.cuda.set_device(cfg['local_rank'])
    
    run_dir = os.path.dirname(args.config)
    figure_dir = os.path.join(run_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    test(cfg, args.model_path, figure_dir)