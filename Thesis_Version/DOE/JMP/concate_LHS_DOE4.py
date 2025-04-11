import os
import pandas as pd

def append_run_data(base_folder, existing_csv, metadata_csv):
    # Load the existing CSV
    existing_df = pd.read_csv(existing_csv)

    # Load the metadata CSV
    metadata_df = pd.read_csv(metadata_csv)
    metadata_df.set_index(metadata_df.columns[0], inplace=True)  # Assume first column has folder names

    # Loop through subfolders
    for i in range(276):
        subfolder = f'DOE4-{i+1}'
        subfolder_path = os.path.join(base_folder, subfolder)
        run_data_path = os.path.join(subfolder_path, 'run_data.csv')

        if os.path.isdir(subfolder_path) and os.path.exists(run_data_path):
            # Load the run data from subfolder
            run_df = pd.read_csv(run_data_path)

            # Find the metadata row matching the folder name
            if subfolder in metadata_df.index:
                meta_row = pd.DataFrame([metadata_df.loc[subfolder]])
                meta_data = pd.concat([meta_row]*run_df.shape[0], ignore_index=True)
                combined_data = pd.concat([run_df, meta_data], axis=1)

                # Append to the existing DataFrame
                existing_df = pd.concat([existing_df, combined_data], ignore_index=True)
            else:
                print(f"Metadata row for folder '{subfolder}' not found. Skipping.")
        else:
            print(f"Folder '{subfolder}' does not exist")
                

    # Save back the updated existing CSV
    existing_df.to_csv('LHS_DO4_Combined.csv', index=False)
    print("Data appended successfully.")



if __name__ == "__main__":
    append_run_data("Architecture/Outputs/DOE4 - Full", "Hypercube_Results.csv", "DOE/DOE_4_Inputs.csv")
