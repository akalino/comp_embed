
# Dataset generation script for Wells Fargo Case Study
# April 2019

# make sure to install these packages before running:
# pip install pandas
# pip install sodapy

import pandas as pd
from sodapy import Socrata


def get_case_study_data():
    """
    Function to match complaint narratives to message ids used in case study

    """
    # Use Socrata client to collect consumer finance data
    client = Socrata("data.consumerfinance.gov", None)
    results = client.get("s6ew-h6mp", limit=99999999)

    # Convert to pandas DataFrame
    results_df = pd.DataFrame.from_records(results)
    results_df = results_df[['complaint_id', 'complaint_what_happened']]
    results_df.rename(columns={'complaint_what_happened':'text'}, inplace=True)
    results_df.complaint_id = results_df.complaint_id.astype('int64')

    # Load Case Study message ids
    case_study_df = pd.read_csv("case_study_msg_ids.csv")
    case_study_df.complaint_id = case_study_df.complaint_id.astype('int64')

    # Join by msg_id
    case_study_df = case_study_df.merge(results_df, on='complaint_id', how='left')
	
	# Drop NAs
    case_study_df.dropna(inplace=True)

    return case_study_df


if __name__ == "__main__":
    # get data
    df = get_case_study_data()

    # write to csv
    df.to_csv('case_study_data.csv', header=True, index=False)

