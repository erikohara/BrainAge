import nibabel
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import customTransforms
from header import *
import monai
import nibabel

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

BATCH_SIZE = 16
N_WORKERS = 4
N_EPOCHS = 15
MAX_IMAGES = 500
LR = 0.001

def setup(rank, world_size):

    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size):

    setup(rank,world_size)

    # Setup DDP:
    # dist.init_process_group("nccl")
    # rank = dist.get_rank()
    # # rank = int(os.environ["LOCAL_RANK"])
    # device = rank % torch.cuda.device_count()
    seed = 1 * dist.get_world_size() + rank
    torch.manual_seed(seed)
    # torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed=1, world_size={dist.get_world_size()}.")

    print(torch.cuda.device_count())

    # Reading the data and the denormalization function
    # images, mean_age, ages, get_age = read_data("data/91", postfix=".nii.gz", max_entries=MAX_IMAGES)
    # images, mean_age, ages, get_age = read_data("/work/forkert_lab/erik/T1_warped", postfix=".nii.gz", max_entries=MAX_IMAGES)

    train_images, train_mean_age, train_ages, train_get_age = read_data("/work/forkert_lab/erik/T1_warped/train", postfix=".nii.gz", max_entries=MAX_IMAGES*.8)
    val_images, val_mean_age, val_ages, val_get_age = read_data("/work/forkert_lab/erik/T1_warped/val", postfix=".nii.gz", max_entries=MAX_IMAGES*.1)
    test_images, test_mean_age, test_ages, test_get_age = read_data("/work/forkert_lab/erik/T1_warped/test", postfix=".nii.gz", max_entries=MAX_IMAGES*.1)


    # Add transforms to the dataset
    # transforms = Compose([monai.transforms.CenterSpatialCrop(roi_size=[150,150]),EnsureChannelFirst(), NormalizeIntensity()])
    transforms = Compose([customTransforms.Crop3D((150,150,150)),EnsureChannelFirst(), NormalizeIntensity()])

    # Define image dataset, data loader
    train_ds = ImageDataset(image_files=train_images, labels=train_ages, dtype=np.float32, transform=transforms,reader="NibabelReader")
    val_ds = ImageDataset(image_files=val_images, labels=val_ages, dtype=np.float32, transform=transforms,reader="NibabelReader")
    test_ds = ImageDataset(image_files=test_images, labels=test_ages, dtype=np.float32, transform=transforms,reader="NibabelReader")

    # Split the data into training and testing sets
    # train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [.8, .1, .1])

    # train_list_eid = [train_ds.dataset.image_files[x] for x in train_ds.indices]
    # val_list_eid = [val_ds.dataset.image_files[x] for x in val_ds.indices]
    # test_list_eid = [test_ds.dataset.image_files[x] for x in test_ds.indices]

    # print(train_list_eid)
    # print(val_list_eid)
    # print(test_list_eid)

    # with open("/home/finn.vamosi/3Brain/split/train.txt", "w") as file:
    #     for name in train_list_eid:
    #         file.write("%s\n" % name)
    #
    # with open("/home/finn.vamosi/3Brain/split/val.txt", "w") as file:
    #     for name in val_list_eid:
    #         file.write("%s\n" % name)
    #
    # with open("/home/finn.vamosi/3Brain/split/test.txt", "w") as file:
    #     for name in test_list_eid:
    #         file.write("%s\n" % name)

    train_loader = DataLoader(train_ds, shuffle=False, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=torch.cuda.is_available(), sampler=DistributedSampler(train_ds))
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=torch.cuda.is_available(), sampler=DistributedSampler(val_ds))
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=torch.cuda.is_available(), sampler=DistributedSampler(test_ds))
    
    # if DEBUG:
    #     print_title("Image Properties")
    #     print(f"Max Tensor Value: {torch.max(ds[0][0])} Min Tensor Value: {torch.min(ds[0][0])}")
    #     print(f"Shape of the image {ds[0][0].shape}")
    #     print_title("Loading the data")
    #     print(f"length of ds: {len(ds)} ")
    #     print(f"Mean Age: {mean_age}")
    #     print_title("Data Splitting")
    #     print(f"Train: {len(train_ds)} Val: {len(val_ds)} Test: {len(test_ds)}")

    # # Check if CUDA is available
    # torch.cuda._lazy_init()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # if device == "cuda":
    #     torch.cuda.empty_cache()
    # if DEBUG:
    #     print("device: ", device)
    
    # Load the model
    # model = DDP(SFCNModel().to(device), device_ids=[rank])
    model=DDP(SFCNModel().to(rank), device_ids=[rank])
    device=rank

    MSELoss_fn = nn.MSELoss()
    MAELoss_fn = nn.L1Loss()
    lr = LR
    opt = torch.optim.Adam(model.parameters(), lr)
    schdlr = torch.optim.lr_scheduler.StepLR(opt, step_size=N_EPOCHS//3, gamma=0.1)
    writer = SummaryWriter()

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
            train_X, train_Y = data[0].to(device) , data[1].to(device)
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
                test_X, test_Y = data[0].to(device) , data[1].to(device)
                test_Y = test_Y.type('torch.cuda.FloatTensor')

                # Make a prediction
                #print("TEST: \n"+test_X)
                pred = model(test_X)

                # Calculate the losses
                MSE_loss = MSELoss_fn(pred, test_Y)
                MAE_loss = MAELoss_fn(pred, test_Y)
                MAE_with_mean_loss = MAE_with_mean_fn(val_mean_age, test_Y)

                MSE_losses.append(MSE_loss.item())
                MAE_losses.append(MAE_loss.item())
                MAE_with_mean_losses.append(MAE_with_mean_loss.item())

                # Display Loss
                pbar2.set_description(f"Epoch {epoch+1:<2} MSE Loss: {MSE_loss.item():<.4f} MAE Loss: {MAE_loss.item():<.4f} Last Predicted Age: {val_get_age(pred[-1].item()):<.4f} Last Actual Age: {val_get_age(test_Y[-1].item()):<.4f}")
        
        # Epoch over
        schdlr.step()

        # Save the model every 10th iteration if the loss is the lowest in this session
        if MSE_loss.item() < min_MSE.item() and epoch % 10 == 0:
            min_MSE = MSE_loss.detach()
            best_metric_epoch = epoch
            # torch.save(model.state_dict(), f'models/epoch_{epoch}_model.pt')
            torch.save(model.state_dict(), '/home/finn.vamosi/3Brain/models/epoch_{epoch}_model.pt')

        writer.add_scalar(f"Training lr={LR}/MSE_train", list_avg(train_losses), epoch)
        writer.add_scalar(f"Testing lr={LR}/MAE_eval", list_avg(MAE_losses), epoch)
        writer.add_scalar(f"Testing lr={LR}/MSE_eval", list_avg(MSE_losses), epoch)
        writer.add_scalar(f"Testing lr={LR}/MAE_with_mean_eval", list_avg(MAE_with_mean_losses), epoch)

    # Training ended
    print_title("End of Training")
    print(f"best metric epoch: {best_metric_epoch}")
    print(f"best metric (MSE): {min_MSE.item()}")

    writer.flush()

    # Saving the model
    # torch.save(model.state_dict(), 'models/end_model.pt')
    torch.save(model.state_dict(), '/home/finn.vamosi/3Brain/models/end_model.pt')

    # Testing
    print_title("Testing")
    model.eval()
    df = pd.DataFrame(columns = ["Age", "Prediction", "ABSError", "ABSMEANError"])
    MSE_losses = []
    MAE_losses = []
    MAE_with_mean_losses = []

    with torch.no_grad():
        pbar3 = tqdm(test_loader)
        for data in pbar3:

            # Extract the input and the labels
            test_X, test_Y = data[0].to(device) , data[1].to(device)
            test_Y = test_Y.type('torch.cuda.FloatTensor')

            # Make a prediction
            pred = model(test_X)

            # Calculate the losses
            MSE_loss = MSELoss_fn(pred, test_Y)
            MAE_loss = MAELoss_fn(pred, test_Y)
            MAE_with_mean_loss = MAE_with_mean_fn(test_mean_age, test_Y)

            MSE_losses.append(MSE_loss.item())
            MAE_losses.append(MAE_loss.item())
            MAE_with_mean_losses.append(MAE_with_mean_loss.item())

            for i, ith_pred in enumerate(pred):
                df.loc[len(df)] = {"Age":test_Y[i].item(), "Prediction":ith_pred.item(), "ABSError": abs(test_Y[i].item() - ith_pred.item()), "ABSMEANError": abs(test_Y[i].item() - test_mean_age)}
    
    # End of testing
    print_title("End of Testing")
    print(f"MAE: {list_avg(MAE_losses)} MSE: {list_avg(MSE_losses)}")

    # Saving predictions into a .csv file
    df.to_csv("/home/finn.vamosi/3Brain/predictions.csv")

    if DEBUG:
        print_title("Testing Data")
        print(df.shape)
        print(df.head())

if __name__ == "__main__":

    if len(sys.argv) > 1: 
        if(sys.argv[1] == '-d'):
            DEBUG = True
    else:
        DEBUG = False
    z=torch.cuda.device_count()    
    print(z)
    mp.spawn(main, args=(z,),nprocs=2)
