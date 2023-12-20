import pickle
from utils import *
from networks import ConvNet, UNet

def main(args):
    if args["num_classes"] < 1:
        raise ValueError("number of classes cannot be zero or negative.")

    H, W = args["size"]

    dataset_directory = "../datasets/oxford-iiit-pet/"

    """ organize real training data """
    train_images_filenames, _, images_directory, masks_directory = get_filenames(dataset_directory)

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

    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        images, masks = images_all[idx_shuffle], masks_all[idx_shuffle]
        return torch.cat((images, masks.unsqueeze(1)), dim=1)
    
    """ initialize synthetic data """
    image_syn = torch.randn(size=(args["num_classes"]*args["ipc"], args["num_channels"]+1, H, W), dtype=torch.float, requires_grad=True, device=args["device"])

    if args["init"] == "real":
        print("initialize synthetic data from random real images")
        for c in range(args["num_classes"]):
            image_syn.data[c*args["ipc"]:(c+1)*args["ipc"]] = get_images(c, args["ipc"]).detach().data
    else:
        print("initialize synthetic data from random noise")

    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    dsa_param = ParamDiffAug()
    mean_tensor = torch.tensor([0.4816, 0.4495, 0.3961])
    std_tensor = torch.tensor([0.2669, 0.2620, 0.2707])

    optimizer_img = torch.optim.SGD([image_syn], lr=args["lr_img"], momentum=0.5)
    optimizer_img.zero_grad()

    soft_label_training = False
    
    metrics = {"dist_loss" : []}
    for d_step in range(1, args["d_steps"]+1):
        # show images
        if args["show_it"] is not None:
            if d_step == 1 or d_step % args["show_it"] == 0:
                plot_image_mask_pairs(image_syn.detach().cpu(), mean_tensor, std_tensor, args)
        
        # save images
        if args["save_it"] is not None:
            if d_step == 1 or d_step % args["save_it"] == 0:
                save_image_mask_pairs(image_syn.detach().cpu(), mean_tensor, std_tensor, args)

        """ training """
        net = ConvNet(channel=args["num_channels"]+1, num_classes=args["num_classes"], net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=(H, W)).to(args["device"])
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False
        
        embed = net.embed

        images_real_all = []
        images_syn_all = []

        loss = torch.tensor(0.0).to(args["device"])

        for c in range(args["num_classes"]):
            img_real = get_images(c, args["real_bsize"])
            img_syn = image_syn[c*args["ipc"]:(c+1)*args["ipc"]].reshape((args["ipc"], args["num_channels"]+1, H, W))

            seed = int(time.time() * 1000) % 100000
            img_real = DiffAugment(img_real, args["dsa_strategy"], seed, dsa_param)
            img_syn = DiffAugment(img_syn, args["dsa_strategy"], seed, dsa_param)

            images_real_all.append(img_real)
            images_syn_all.append(img_syn)

        images_real_all = torch.cat(images_real_all, dim=0)
        images_syn_all = torch.cat(images_syn_all, dim=0)

        output_real = embed(images_real_all).detach()
        output_syn = embed(images_syn_all)

        loss += torch.sum((torch.mean(output_real.reshape(args["num_classes"], args["real_bsize"], -1), dim=1) - torch.mean(output_syn.reshape(args["num_classes"], args["ipc"], -1), dim=1))**2)

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()

        if not soft_label_training:
            soft_label_training = True

        if args["show_it"] is not None: 
            if d_step % args["show_it"] == 0:
                    print(loss.item())

        metrics["dist_loss"].append(loss.item())

    return image_syn, metrics


if __name__ == "__main__":
    args = {
        "eval_it" : 10000,
        "show_it" : None,
        "save_it" : 5000,
        "save_dir" : "",

        "num_classes" : 1, # excluding background, e.g., 1 is binary segmentation, 2 is semantic segmentation
        "ipc" : 50,

        "init" : "real",
        "lr_img" : 1,

        "size" : (64, 64), 
        "num_channels" : 3,

        "d_steps" : 20000,

        "real_bsize" : 256,
        "eval_batch_size" : 256,
        
        "eval_lr" : 0.01,
        "eval_epochs" : 200,

        "early_stopping" : False,

        "dsa_strategy" : "color_crop_cutout_flip_scale_rotate",

        "device" : "cuda" if torch.cuda.is_available() else "cpu"
    }
    image_syn, metrics = main(args)

    with open(os.path.join(args["save_dir"], "my_dict.pkl"), "wb") as pickle_file:
        pickle.dump(metrics, pickle_file)
