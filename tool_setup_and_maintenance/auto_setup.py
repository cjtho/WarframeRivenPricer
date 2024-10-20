from warframe_marketplace_predictor import filepaths
from warframe_marketplace_predictor.tool_setup_and_maintenance import setup_weapon_information, download_data, \
    create_marketplace_dataframe
from warframe_marketplace_predictor.training.trainers import train_price_model


def main(run_full_pipeline=False,
         download_onwards=False,
         dataframe_onwards=False,
         train_onwards=False,
         weapon_ranks_onwards=False,
         overwrite_marketplace=False):
    def run_pipeline(verify_file_paths=False,
                     download_data_files=False,
                     create_dataframe=False,
                     train_model=False,
                     setup_weapon_ranks=False):
        if verify_file_paths:
            print("Verifying file paths...")
            filepaths.main()
            print("File paths verified.")

        if download_data_files:
            print("Downloading all data... (This may take approximately 15 minutes)")
            if overwrite_marketplace is True:
                if (input("WARNING: You are about to delete and replace your marketplace data. Type 'YES' to confirm ")
                        == "YES"):
                    download_data.main(running_all=True, overwrite_marketplace_data=True)
            else:
                download_data.main(running_all=True, overwrite_marketplace_data=False)
            print("Marketplace data downloaded successfully.")

        if create_dataframe:
            print("Creating training dataframe...")
            create_marketplace_dataframe.main()
            print("Dataframe created successfully.")

        if train_model:
            print("Training the model... (This may take approximately 5 minutes)")
            train_price_model.train_model(show_graph=False)
            print("Model training completed.")

        if setup_weapon_ranks:
            print("Setting up weapon ranks... (This may take approximately 20 minutes)")
            setup_weapon_information.main()
            print("Weapon ranks setup complete.")

    # Define scenarios based on input flags
    if run_full_pipeline:
        run_pipeline(verify_file_paths=True,
                     download_data_files=True,
                     create_dataframe=True,
                     train_model=True,
                     setup_weapon_ranks=True)
    elif download_onwards:
        run_pipeline(download_data_files=True, create_dataframe=True, train_model=True, setup_weapon_ranks=True)
    elif dataframe_onwards:
        run_pipeline(create_dataframe=True, train_model=True, setup_weapon_ranks=True)
    elif train_onwards:
        run_pipeline(train_model=True, setup_weapon_ranks=True)
    elif weapon_ranks_onwards:
        run_pipeline(setup_weapon_ranks=True)
    else:
        print("No valid scenario selected.")
        return

    print("Setup complete. You may now navigate to 'rivens_analysis',"
          " scroll to the bottom, input your rivens, and run the script.")


if __name__ == "__main__":
    main(
        run_full_pipeline=True,

        download_onwards=False,
        dataframe_onwards=False,
        train_onwards=False,
        weapon_ranks_onwards=False,

        overwrite_marketplace=True
    )
