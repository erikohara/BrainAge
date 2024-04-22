import math

import customTransforms
from SFCN import SFCNModelMONAI
from header_train import *
from train import cwd

BATCH_SIZE = 1
N_WORKERS = 4
MAX_IMAGES = -1


def main():
    """
    Tests on original images in the test set whichs have counterfactuals
    """
    train_images, val_images, test_images, mean_age, ages_train, ages_val, ages_test, get_age = read_data("/work/forkert_lab/erik/T1_warped",
                                                                               postfix=".nii.gz",
                                                                               max_entries=MAX_IMAGES)

    # Add transforms to the dataset
    transforms = Compose([customTransforms.Crop3D((150, 150, 100)), EnsureChannelFirst(), NormalizeIntensity()])

    # Define image dataset, data loader
    train_ds = ImageDataset(image_files=train_images, labels=ages_train, dtype=np.float32, transform=transforms,
                            reader="NibabelReader")
    val_ds = ImageDataset(image_files=val_images, labels=ages_val, dtype=np.float32, transform=transforms,
                          reader="NibabelReader")
    test_ds = ImageDataset(image_files=test_images, labels=ages_test, dtype=np.float32, transform=transforms,
                           reader="NibabelReader")

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                              pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                            pin_memory=torch.cuda.is_available(), drop_last=True)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE, num_workers=N_WORKERS,
                             pin_memory=torch.cuda.is_available(), drop_last=True)

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


    with torch.no_grad():
        print_title("Traing set")
        df_train= pd.DataFrame(columns=["EID", "Age", "Prediction", "ABSError", "ABSMEANError"])
        MSE_losses = []
        MAE_losses = []
        MAE_with_mean_losses = []
        idx = 0
        imgs = train_ds.image_files
        pbar1 = tqdm(train_loader)
        for data in pbar1:

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
                    df_train.loc[len(df_train)] = {"EID": img.split('/')[-1].split('.')[0], "Age": test_Y[i].item(),
                                       "Prediction": ith_pred.item(),
                                       "ABSError": abs(test_Y[i].item() - ith_pred.item()),
                                       "ABSMEANError": abs(test_Y[i].item() - mean_age)}

        # End of testing
        print_title("End of Training set")
        print(f"ORIGINAL TRAINING\nMAE: {list_avg(MAE_losses)} MSE: {list_avg(MSE_losses)}")

        # Saving predictions into a .csv file
        df_train.to_csv(f"{cwd}predictions_real_train.csv")
    
        print_title("Validation set")
        df_val = pd.DataFrame(columns=["EID", "Age", "Prediction", "ABSError", "ABSMEANError"])
        MSE_losses = []
        MAE_losses = []
        MAE_with_mean_losses = []

        idx = 0
        imgs = val_ds.image_files

        pbar2 = tqdm(val_loader)
        for data in pbar2:

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
                    df_val.loc[len(df_val)] = {"EID": img.split('/')[-1].split('.')[0], "Age": test_Y[i].item(),
                                       "Prediction": ith_pred.item(),
                                       "ABSError": abs(test_Y[i].item() - ith_pred.item()),
                                       "ABSMEANError": abs(test_Y[i].item() - mean_age)}

        # End of testing
        print_title("End of validation")
        print(f"ORIGINAL Validation\nMAE: {list_avg(MAE_losses)} MSE: {list_avg(MSE_losses)}")

        # Saving predictions into a .csv file
        df_val.to_csv(f"{cwd}predictions_real_val.csv")
        
        print_title("Test set")
        df_test = pd.DataFrame(columns=["EID", "Age", "Prediction", "ABSError", "ABSMEANError"])
        MSE_losses = []
        MAE_losses = []
        MAE_with_mean_losses = []
        idx = 0
        imgs = test_ds.image_files
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
                    df_test.loc[len(df_test)] = {"EID": img.split('/')[-1].split('.')[0], "Age": test_Y[i].item(),
                                       "Prediction": ith_pred.item(),
                                       "ABSError": abs(test_Y[i].item() - ith_pred.item()),
                                       "ABSMEANError": abs(test_Y[i].item() - mean_age)}

        # End of testing
        print_title("End of Testing")
        print(f"ORIGINAL TEST\nMAE: {list_avg(MAE_losses)} MSE: {list_avg(MSE_losses)}")

        # Saving predictions into a .csv file
        df_test.to_csv(f"{cwd}predictions_real_test.csv")

    if DEBUG:
        print_title("Testing Data")
        print(df_test.shape)
        print(df_test.head())


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if (sys.argv[1] == '-d'):
            DEBUG = True
    else:
        DEBUG = False
    main()
