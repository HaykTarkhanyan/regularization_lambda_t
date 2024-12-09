import os
import json
import pandas as pd
from PIL import Image
import streamlit as st

# Define the base directory structure
BASE_DIR = "to_overleaf"

# List the options for regularization types and dataset groups
regularization_types = ["L1", "L2", "elastic"]
dataset_groups = ["3x_1 - 2x_2", "optim_slides", "diabetes", "california_housing"]

# Create sidebar options
st.sidebar.header("Experiment")
selected_group = st.sidebar.selectbox("Select Dataset", dataset_groups)
selected_regularization = st.sidebar.selectbox("Select Regularization Type", regularization_types)

# Construct the folder path based on user selection
folder_name = f"{selected_group}_{selected_regularization}"#.replace(" ", "_").replace("-", "_")
folder_path = os.path.join(BASE_DIR, folder_name)

# Display the selected folder
st.header(f"Relationship between lambda values and L1 and L2 norms for {selected_group} dataset (for {selected_regularization} regularization)")

# Check if the folder exists
if os.path.exists(folder_path):
    report_path = os.path.join(folder_path, "report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as report_file:
            report = json.load(report_file)
    else:
        st.warning("No report.json file found in this folder.")

    st.subheader("Details:")
    
    st.write("Number of samples:", report["N_SAMPLES"])
    st.write("Number of features:", report["N_FEATURES"])
    st.write(f"Used `{report['NUM_LAMBDAS']}` lambda values between `10^{report['MIN_LAMBDA_LOG_10']}` and `10^{report['MAX_LAMBDA_LOG_10']}`")
    
    st.subheader("Figures:")
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    
    file_name_to_title = {
        "lambda_combined_norm_rel": "L1 and L2 Norms for full range of lambda values",
        "lambda_combined_norm_rel_log_lambda": "L1 and L2 Norms for full range of lambda values (log scale)",
        "lambda_combined_norm_rel_narrow_x": "L1 and L2 Norms for lambdas from 0.1 to 10",
        "lambda_combined_norm_rel_super_narrow": "L1 and L2 Norms for lambdas from 0.1 to 1",
    }
    
    if image_files:
        # Display images in a 2x2 grid
        cols = st.columns(2)
        for idx, image_file in enumerate(sorted(image_files)):  # Sort to ensure consistent order
            col = cols[idx % 2]
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            
            key_for_title = image_file.replace(".png", "").split("\\")[-1]
            with col:
                st.image(image, caption=file_name_to_title[key_for_title], use_container_width =True)
    else:
        st.warning("No images found in this folder.")
    
    st.subheader("Relationship type estimation:")
    def process_table(df):
        """Process the DataFrame to rename columns and round values."""
        df = df.T  # Transpose the DataFrame
        df = df.rename(columns={"p_val": "P Value", "pearson": "Pearson", 
                                "error": "Error"})  # Rename 'p_val' to 'P Value'
        df = df.round(2)  # Round all numeric values to two decimal places
        df = df.sort_values("R2", ascending=False)  # Sort by R2 in descending order
        return df

    l1_norm = process_table(pd.DataFrame(report["relationship_l1_norm"]))
    l2_norm = process_table(pd.DataFrame(report["relationship_l2_norm"]))

    st.write("L1 Norm:")
    st.write(l1_norm)
    st.write("L2 Norm:")
    st.write(l2_norm)
else:
    st.warning("No folder found for the selected dataset and regularization type.")
    st.warning(f"you chose {selected_group} and {selected_regularization}")