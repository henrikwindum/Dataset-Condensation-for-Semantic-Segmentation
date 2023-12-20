from utils import (
    train, 
    validate,
    get_filenames,
    get_dataloader_from_filenames,
    STANDARD_TRANSFORM,
    SHIFT_SCALE_ROTATE_TRANSFORM,
)

from networks import UNet 

import os
import torch
import torch.nn as nn

def main(args):
    dataset_directory = "../datasets/oxford-iiit-pet"
    train_images_filenames, val_images_filenames, images_directory, masks_directory = get_filenames(dataset_directory)
    
    if args["save_dir"]:
        os.makedirs(args["save_dir"], exist_ok=True)

    train_transform = SHIFT_SCALE_ROTATE_TRANSFORM(args)
    val_transform = STANDARD_TRANSFORM(args)

    train_loader = get_dataloader_from_filenames(train_images_filenames, images_directory, masks_directory, train_transform, batch_size=args["batch_size"], shuffle=True)
    val_loader = get_dataloader_from_filenames(val_images_filenames, images_directory, masks_directory, val_transform, batch_size=args["batch_size"], shuffle=False)

    criterion = nn.BCELoss().to(args["device"])

    for num_expert in range(1, args["num_experts"]+1):
        print("INITIALIZING NEW UNET MODEL")
        model = UNet(3, 1).to(args["device"])
        optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"])

        for epoch in range(1, args["num_epochs"]+1):
            model = train(train_loader, model, criterion, optimizer, epoch, args, retval_model=True)
            if args["validate"]:
                _, _ = validate(val_loader, model, criterion, epoch, args)

            if args["save_dir"]:
                torch.save(model.state_dict(), os.path.join(args["save_dir"], "model{}_e{}".format(num_expert, epoch)))

        if args["save_dir"]:
            torch.save(model.state_dict(), os.path.join(args["save_dir"], "full_model{}".format(num_expert)))

if __name__ == "__main__":
    args = {
        "save_dir" : "",
        
        "size" : (128, 128),

        "num_experts" : 20,

        "num_epochs" : 100,
        "batch_size" : 128,
        "lr" : 0.01,

        "validate" : False,

        "device" : "cuda" if torch.cuda.is_available() else "cpu"
    }

    main(args)