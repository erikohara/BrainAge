import math

import customTransforms
from SFCN import SFCNModelMONAI
from header_cfs import *
from train import cwd

BATCH_SIZE = 1
N_WORKERS = 4
MAX_IMAGES = -1

def main():
    """
    Tests on counterfactual images
    """
    images, mean_age, ages, get_age = read_data("/work/forkert_lab/erik/MACAW/cf_images/PCA_post",
                                                postfix=".nii.gz",
                                                max_entries=MAX_IMAGES)

    # Add transforms to the dataset
    transforms = Compose([customTransforms.Crop3D((150, 150, 100)), EnsureChannelFirst(), NormalizeIntensity()])

    # Define image dataset, data loader
    test_ds = ImageDataset(image_files=images, labels=ages, dtype=np.float32, transform=transforms,
                           reader="NibabelReader")

    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                             pin_memory=torch.cuda.is_available())

    # Check if CUDA is available
    torch.cuda._lazy_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if device == "cuda":
        torch.cuda.empty_cache()
    if DEBUG:
        print("device: ", device)

    model = SFCNModelMONAI().to(device)
    model.load_state_dict(torch.load(f"{cwd}models/end_model.pt"))

    MSELoss_fn = nn.MSELoss()
    MAELoss_fn = nn.L1Loss()

    # Testing
    print_title("Testing")
    model.eval()
    df = pd.DataFrame(columns=["EID", "Age", "Prediction", "ABSError", "ABSMEANError"])
    MSE_losses = []
    MAE_losses = []
    MAE_with_mean_losses = []

    idx = 0
    imgs = test_ds.image_files

    with torch.no_grad():
        pbar3 = tqdm(test_loader)
        for data in pbar3:

            img = imgs[idx]
            idx += 1

            # Extract the input and the labels
            test_X, test_Y = data[0].to(device), data[1].to(device)
            test_Y = test_Y.type('torch.cuda.FloatTensor')

            # Make a prediction
            pred = model(test_X)

            if not math.isnan(pred):

                # Calculate the losses
                MSE_loss = MSELoss_fn(pred, test_Y)
                MAE_loss = MAELoss_fn(pred, test_Y)
                MAE_with_mean_loss = MAE_with_mean_fn(mean_age, test_Y)

                # print(pred, test_Y, MSE_loss, MAE_loss)

                MSE_losses.append(MSE_loss.item())
                MAE_losses.append(MAE_loss.item())
                MAE_with_mean_losses.append(MAE_with_mean_loss.item())

                for i, ith_pred in enumerate(pred):
                    df.loc[len(df)] = {"EID": img.split('/')[-1].split('_')[0], "Age": test_Y[i].item(),
                                       "Prediction": ith_pred.item(),
                                       "ABSError": abs(test_Y[i].item() - ith_pred.item()),
                                       "ABSMEANError": abs(test_Y[i].item() - mean_age)}

    # End of testing
    print_title("End of Testing")
    print(f"COUNTERFACTUAL TEST\nMAE: {list_avg(MAE_losses)} MSE: {list_avg(MSE_losses)}")

    # Saving predictions into a .csv file
    df.to_csv(f"{cwd}predictions_cf.csv")

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
    main()
