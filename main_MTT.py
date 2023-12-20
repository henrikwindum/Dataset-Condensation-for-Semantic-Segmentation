from utils import (
    get_data,
    get_filenames, 
    plot_image_mask_pairs, 
    save_image_mask_pairs,
    preprocess_experts, 
)

import copy
import numpy as np

import torch
import torch.nn as nn
from networks import UNet
from ReparamModule import ReparamModule

def main(args):
    dataset_directory = "../datasets/oxford-iiit-pet/"

    """ organize real training data """
    train_images_filenames, _, images_directory, masks_directory = get_filenames(dataset_directory)

    H, W = args["size"]

    data = {
            "images" : torch.empty(size=(len(train_images_filenames), args["num_channels"], H, W), dtype=torch.float),
            "masks" : torch.empty(size=(len(train_images_filenames), H, W), dtype=torch.long),
            "labels" : torch.empty(size=(len(train_images_filenames), ), dtype=torch.long)
    }
    data = get_data(data, train_images_filenames, images_directory, masks_directory, args)

    images_all = data["images"].to(args["device"])
    labels_all = data["labels"].to(args["device"])
    masks_all = data["masks"].to(args["device"])

    indices_class = [[] for _ in range(args["num_classes"])]
    for i, label in enumerate(labels_all):
        indices_class[label].append(i)

    def get_images_masks(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle], masks_all[idx_shuffle].unsqueeze(1)

    image_syn = torch.randn(size=(args["num_classes"]*args["ipc"], args["num_channels"], H, W), dtype=torch.float, device=args["device"]).requires_grad_(True)
    mask_syn = torch.rand(size=(args["num_classes"]*args["ipc"], 1, H, W), dtype=torch.float, device=args["device"]).requires_grad_(True)
    lr_syn = torch.tensor(args["lr_syn"], dtype=torch.float32, device=args["device"]).requires_grad_(True)

    if args["init"] == "real":
            print("initialize synthetic data from random real images")
            for c in range(args["num_classes"]):
                images, masks = get_images_masks(c, args["ipc"])
                image_syn.data[c*args["ipc"]:(c+1)*args["ipc"]] = images.detach().data
                mask_syn.data[c*args["ipc"]:(c+1)*args["ipc"]] = masks.detach().data
    else:
        print("initialize synthetic data from random noise")

    if args["fixed_mask"]:
        optimizer_img = torch.optim.SGD([image_syn], lr=args["lr_img"], momentum=0.5)
        mask_syn = mask_syn.detach()
    else: 
        optimizer_img = torch.optim.SGD([image_syn, mask_syn], lr=args["lr_img"], momentum=0.5)
    optimizer_lr = torch.optim.SGD([lr_syn], lr=args["lr_lr"], momentum=0.5)

    optimizer_img.zero_grad()
    optimizer_lr.zero_grad()

    criterion = nn.BCEWithLogitsLoss()

    mean_tensor = torch.tensor([0.4816, 0.4495, 0.3961])
    std_tensor = torch.tensor([0.2669, 0.2620, 0.2707])

    experts_directory = "../code/experts_directory"
    experts_filenames = preprocess_experts(experts_directory)

    syn_images_save = {}
    for d_step in range(1, args["d_steps"] + 1):
        """ show images """
        if args["show_it"] is not None:
            if d_step == 1 or d_step % args["show_it"] == 0:
                plot_image_mask_pairs(copy.deepcopy(torch.cat((image_syn, mask_syn), dim=1).detach().cpu()), mean_tensor, std_tensor, args)

        """ save and store images """
        if args["save_it"] is not None:
            if d_step == 1 or d_step % args["save_it"] == 0:
                image_mask_syn_save = copy.deepcopy(torch.cat((image_syn, mask_syn), dim=1).detach().cpu())
                save_image_mask_pairs(image_mask_syn_save, mean_tensor, std_tensor, args)
                syn_images_save[d_step] = image_mask_syn_save

        num_expert = np.random.randint(1, args["e_num"] + 1)
        init_epoch = np.random.randint(0, args["max_starting_epoch"])

        expert_params = torch.load(experts_filenames[num_expert][init_epoch])
        target_params = torch.load(experts_filenames[num_expert][init_epoch + args["target_epoch"]])

        student_net = UNet(in_channels=3, out_channels=1).to(args["device"])
        reparam_nps = {np for np, _ in student_net.named_parameters()}
        student_net = ReparamModule(student_net)

        starting_params = torch.cat([p.data.to(args["device"]).reshape(-1) for n, p in expert_params.items() if n in reparam_nps], 0)

        student_params = [torch.cat([p.data.to(args["device"]).reshape(-1) for n, p in expert_params.items() if n in reparam_nps], 0).requires_grad_(True)]

        target_params = torch.cat([p.data.to(args["device"]).reshape(-1) for n, p in target_params.items() if n in reparam_nps], 0)

        student_net.train()

        indices_chunks = []
        for epoch in range(args["student_epochs"]):
            if not indices_chunks:
                indices = torch.randperm(len(image_syn))
                indices_chunks = list(torch.split(indices, args["student_batch_size"]))
            
            indices_chunk = indices_chunks.pop()

            images = image_syn[indices_chunk]
            target = mask_syn[indices_chunk]

            output = student_net(images, flat_param=student_params[-1])

            loss = criterion(output, target)
            grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - lr_syn * grad)

        param_loss = torch.tensor(0.0).to(args["device"])
        param_dist = torch.tensor(0.0).to(args["device"])

        param_loss += nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += nn.functional.mse_loss(starting_params, target_params, reduction="sum")
        param_loss /= param_dist

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        param_loss.backward()

        optimizer_img.step()
        optimizer_lr.step() 

        del student_params

        if d_step % 50 == 0:
            print("Distillation Step: {:>4}, Expert No. {:>2}, t = {:>2} Loss: {:.4f}, Synthetic Learning Rate: {:.6}"
                .format(d_step, num_expert, init_epoch, param_loss.item(), lr_syn.item()))
    return syn_images_save

if __name__ == "__main__":
    args = {
        "show_it" : None,
        "save_it" : 1000,
        "save_dir" : "",

        "fixed_mask" : True,
        "size" : (128, 128), 
        "init" : "real",
        "ipc" : 50,

        "d_steps" : 20000,
        "lr_img" : 1000,
        "lr_syn" : 0.01,
        "lr_lr" : 1e-7,

        "e_num" : 29,
        "target_epoch" : 3,
        "max_starting_epoch" : 25,

        "student_epochs" : 10,
        "student_batch_size" : 1,

        "num_channels" : 3, # could potentionally be stored with respective dataset
        "num_classes" : 1, # excluding background, e.g., 1 is binary segmentation, 2 is semantic segmentation

        "device" : "cuda" if torch.cuda.is_available() else "cpu"
    }

    syn_images = main(args)