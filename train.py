from torch.utils.data import DistributedSampler
#from torch.utils.tensorboard import SummaryWriter

import customTransforms
from SFCN import SFCNModelMONAI
from header_train import *
#import monai
#import nibabel

import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.multiprocessing as mp

BATCH_SIZE = 8
N_WORKERS = 4
N_EPOCHS = 20
MAX_IMAGES = 6000
LR = 0.0001
CKPT_EVERY = 999
USE_CKPT = False
CKPT_NUM = 3

cwd = "/home/erik.ohara/BrainAge/"

def setup(rank, world_size):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main():
    """
    Trains an age prediction model on 3D brain scan images
    """
    # setup(rank, world_size)

    # Setup DDP:
    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    # # rank = int(os.environ["LOCAL_RANK"])
    # device = rank % torch.cuda.device_count()
    # seed = 1 * dist.get_world_size() + rank
    # torch.manual_seed(seed)
    # # torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, seed=1, world_size={dist.get_world_size()}.")
    # print(torch.cuda.device_count())

    # Load the images
    train_images, val_images, test_images, mean_age, ages, get_age = read_data("/work/forkert_lab/erik/T1_warped",
                                                                               postfix=".nii.gz",
                                                                               max_entries=MAX_IMAGES)

    # Add transforms to the dataset
    transforms = Compose([customTransforms.Crop3D((150, 150, 100)), EnsureChannelFirst(), NormalizeIntensity()])

    # Define image dataset, data loader
    train_ds = ImageDataset(image_files=train_images, labels=ages, dtype=np.float32, transform=transforms,
                            reader="NibabelReader")
    val_ds = ImageDataset(image_files=val_images, labels=ages, dtype=np.float32, transform=transforms,
                          reader="NibabelReader")
    test_ds = ImageDataset(image_files=test_images, labels=ages, dtype=np.float32, transform=transforms,
                           reader="NibabelReader")

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                            pin_memory=torch.cuda.is_available(), drop_last=True)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                             pin_memory=torch.cuda.is_available(), drop_last=True)

    # if DEBUG:
    #     print_title("Image Properties")
    #     print(f"Max Tensor Value: {torch.max(ds[0][0])} Min Tensor Value: {torch.min(ds[0][0])}")
    #     print(f"Shape of the image {ds[0][0].shape}")
    #     print_title("Loading the data")
    #     print(f"length of ds: {len(ds)} ")
    #     print(f"Mean Age: {mean_age}")
    #     print_title("Data Splitting")
    #     print(f"Train: {len(train_ds)} Val: {len(val_ds)} Test: {len(test_ds)}")

    # Check if CUDA is available
    torch.cuda._lazy_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device == "cuda":
        torch.cuda.empty_cache()
    if DEBUG:
        print("device: ", device)

    lr = LR
    if USE_CKPT:
        print("Using checkpoint")
        ckpt = torch.load(f"{cwd}checkpoints/{CKPT_NUM:07d}.pt")
        model = ckpt.load_state_dict(ckpt['model'])
        opt = ckpt.load_state_dict(ckpt['optimizer'])
    else:
        print("Not using checkpoint")
        model = SFCNModelMONAI().to(device)
        opt = torch.optim.Adam(model.parameters(), lr)

    MSELoss_fn = nn.MSELoss()
    MAELoss_fn = nn.L1Loss()
    schdlr = torch.optim.lr_scheduler.StepLR(opt, step_size=N_EPOCHS // 3, gamma=0.1)
    #writer = SummaryWriter()

    # Training the model
    print_title("Training")
    min_MSE = torch.tensor(float("inf")).to(device)
    best_metric_epoch = -1
    ep_pbar = tqdm(range(N_EPOCHS))
    for epoch in ep_pbar:
        model.train()
        train_losses = []

        pbar = tqdm(train_loader)
        for data in pbar:
            # Extract the input and the labels
            train_X, train_Y = data[0].to(device), data[1].to(device)
            train_Y = train_Y.type('torch.cuda.FloatTensor')

            # Zero the gradient
            opt.zero_grad()

            # Make a prediction
            pred = model(train_X)

            # Calculate the loss and backpropagate
            loss = MSELoss_fn(pred, train_Y)
            loss.backward()

            # Adjust the learning weights
            opt.step()

            # Calculate stats
            train_losses.append(loss.item())
            pbar.set_description(f"######## Training Loss: {loss.item():<.6f} ")

        # Validation
        model.eval()
        MSE_losses = []
        MAE_losses = []
        MAE_with_mean_losses = []

        with torch.no_grad():
            pbar2 = tqdm(val_loader)
            for data in pbar2:
                # Extract the input and the labels
                test_X, test_Y = data[0].to(device), data[1].to(device)
                test_Y = test_Y.type('torch.cuda.FloatTensor')

                # Make a prediction
                pred = model(test_X)

                # Calculate the losses
                MSE_loss = MSELoss_fn(pred, test_Y)
                MAE_loss = MAELoss_fn(pred, test_Y)
                MAE_with_mean_loss = MAE_with_mean_fn(mean_age, test_Y)

                MSE_losses.append(MSE_loss.item())
                MAE_losses.append(MAE_loss.item())
                MAE_with_mean_losses.append(MAE_with_mean_loss.item())

                # Display Loss
                pbar2.set_description(
                    f"Epoch {epoch + 1:<2} MSE Loss: {MSE_loss.item():<.4f} MAE Loss: {MAE_loss.item():<.4f} "
                    f"Last Predicted Age: {get_age(pred[-1].item()):<.4f} Last Actual Age: {get_age(test_Y[-1].item()):<.4f}")

        if epoch % CKPT_EVERY == 0 and epoch > 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": opt.state_dict()
            }
            checkpoint_path = f"{cwd}checkpoints/{epoch:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            #writer.add_text(f"Saved checkpoint to {checkpoint_path}")

        # Epoch over
        schdlr.step()
        # schdlr.step(MSELoss_fn())

        # Save the model every 10th iteration if the loss is the lowest in this session
        if MSE_loss.item() < min_MSE.item() and epoch % 5 == 0:
            min_MSE = MSE_loss.detach()
            best_metric_epoch = epoch
            # torch.save(model.state_dict(), f'models/epoch_{epoch}_model.pt')
            torch.save(model.state_dict(), f'{cwd}models/epoch_{epoch}_model.pt')

        #writer.add_scalar(f"Training lr={LR}/MSE_train", list_avg(train_losses), epoch)
        #writer.add_scalar(f"Validation lr={LR}/MAE_eval", list_avg(MAE_losses), epoch)
        #writer.add_scalar(f"Validation lr={LR}/MSE_eval", list_avg(MSE_losses), epoch)
        #writer.add_scalar(f"Validation lr={LR}/MAE_with_mean_eval", list_avg(MAE_with_mean_losses), epoch)

    # Training ended
    print_title("End of Training")
    print(f"best metric epoch: {best_metric_epoch}")
    print(f"best metric (MSE): {min_MSE.item()}")

    #writer.flush()

    # Saving the model
    # torch.save(model.state_dict(), 'models/end_model.pt')
    torch.save(model.state_dict(), f'{cwd}models/end_model.pt')

    # Testing
    print_title("Testing")
    model.eval()
    df = pd.DataFrame(columns=["Age", "Prediction", "ABSError", "ABSMEANError"])
    MSE_losses = []
    MAE_losses = []
    MAE_with_mean_losses = []

    with torch.no_grad():
        pbar3 = tqdm(test_loader)
        for data in pbar3:

            # Extract the input and the labels
            test_X, test_Y = data[0].to(device), data[1].to(device)
            test_Y = test_Y.type('torch.cuda.FloatTensor')

            # Make a prediction
            pred = model(test_X)

            # Calculate the losses
            MSE_loss = MSELoss_fn(pred, test_Y)
            MAE_loss = MAELoss_fn(pred, test_Y)
            MAE_with_mean_loss = MAE_with_mean_fn(mean_age, test_Y)

            MSE_losses.append(MSE_loss.item())
            MAE_losses.append(MAE_loss.item())
            MAE_with_mean_losses.append(MAE_with_mean_loss.item())

            for i, ith_pred in enumerate(pred):
                df.loc[len(df)] = {"Age": test_Y[i].item(), "Prediction": ith_pred.item(),
                                   "ABSError": abs(test_Y[i].item() - ith_pred.item()),
                                   "ABSMEANError": abs(test_Y[i].item() - mean_age)}

    # End of testing
    print_title("End of Testing")
    print(f"MAE: {list_avg(MAE_losses)} MSE: {list_avg(MSE_losses)}\n\nWARNING: "
          f"This returns a much higher error than it should. Run test_all.py to get a more accurate measure.\n\n")

    # Saving predictions into a .csv file
    df.to_csv(f"{cwd}predictions.csv")

    if DEBUG:
        print_title("Testing Data")
        print(df.shape)
        print(df.head())


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if (sys.argv[1] == '-d'):
            DEBUG = True
    else:
        DEBUG = False
    # z = torch.cuda.device_count()
    # print(z)
    # mp.spawn(main, args=(z,), nprocs=1)
    main()
