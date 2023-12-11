#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df3=pd.read_csv("C://Users//mahes//MapUp-Data-Assessment-F//datasets//dataset-3.csv")

df3.head()


# In[4]:


def calculate_distance_matrix(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create a DataFrame for distances
    distance_df = pd.DataFrame(index=df['id_start'].unique(), columns=df['id_end'].unique())

    # Populate the DataFrame with cumulative distances
    for _, row in df.iterrows():
        distance_df.loc[row['id_start'], row['id_end']] = row['distance']

    # Convert the DataFrame to numeric values
    distance_df = distance_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Ensure the matrix is symmetric
    distance_matrix = distance_df + distance_df.T

    # Set diagonal values to 0
    np.fill_diagonal(distance_matrix.values, 0)

    return distance_matrix

# Example usage
file_path = "C://Users//mahes//MapUp-Data-Assessment-F//datasets//dataset-3.csv"
result_df = calculate_distance_matrix(file_path)
print(result_df)


# In[ ]:





# In[5]:



def unroll_distance_matrix(distance_matrix):
    # Extract upper triangular part of the distance matrix
    upper_triangular = np.triu(distance_matrix.values, k=1)

    # Get row and column indices for non-zero values in the upper triangular part
    row_indices, col_indices = np.where(upper_triangular != 0)

    # Create a DataFrame with id_start, id_end, and distance columns
    unrolled_df = pd.DataFrame({
        'id_start': distance_matrix.index[row_indices],
        'id_end': distance_matrix.columns[col_indices],
        'distance': upper_triangular[row_indices, col_indices]
    })

    return unrolled_df

# Example usage
file_path = "C://Users//mahes//MapUp-Data-Assessment-F//datasets//dataset-3.csv"
distance_matrix = calculate_distance_matrix(file_path)  # Assuming you have a function that calculates distance matrix
result_df = unroll_distance_matrix(distance_matrix)
print(result_df)


# In[ ]:





# In[6]:


def find_ids_within_ten_percentage_threshold(distance_df, reference_value):
    # Filter rows with the given reference value in the 'id_start' column
    reference_rows = distance_df[distance_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_rows['distance'].mean()

    # Calculate the lower and upper bounds within 10% threshold
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1

    # Filter rows within the 10% threshold
    within_threshold = distance_df[
        (distance_df['distance'] >= lower_bound) &
        (distance_df['distance'] <= upper_bound)
    ]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids

# Example usage
file_path = "C://Users//mahes//MapUp-Data-Assessment-F//datasets//dataset-3.csv"
distance_matrix = calculate_distance_matrix(file_path)  
unrolled_df = unroll_distance_matrix(distance_matrix)   

# Choose a reference value (replace with an actual value from your dataset)
reference_value = 1

result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)
print(result_ids)
# it indicates 0


# In[ ]:





# In[7]:


def calculate_toll_rate(distance_df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        distance_df[vehicle_type] = distance_df['distance'] * rate_coefficient

    return distance_df

# Example usage
file_path = "C://Users//mahes//MapUp-Data-Assessment-F//datasets//dataset-3.csv"
distance_matrix = calculate_distance_matrix(file_path)  # Assuming you have a function that calculates distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)   # Assuming you have a function that unrolls the distance matrix
toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)


# In[ ]:





# In[8]:


import datetime

def calculate_time_based_toll_rates(toll_rate_df):
    # Define time ranges and discount factors for weekdays and weekends
    weekday_time_ranges = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)),
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59))
    ]

    weekend_time_range = (datetime.time(0, 0, 0), datetime.time(23, 59, 59))
    weekend_discount_factor = 0.7

    # Create lists to store calculated values
    start_day_list, start_time_list, end_day_list, end_time_list, discount_factor_list = [], [], [], [], []

    # Iterate through each unique (id_start, id_end) pair
    for idx, group in toll_rate_df.groupby(['id_start', 'id_end']):
        id_start, id_end = idx

        # Iterate through each day of the week
        for day in range(7):
            # Get the corresponding day name
            day_name = datetime.date(2023, 1, 2 + day).strftime("%A")  # January 2, 2023 is a Monday

            # Apply discount factor based on weekdays or weekends
            if day < 5:  # Weekdays
                for start_time, end_time in weekday_time_ranges:
                    start_day_list.extend([day_name] * len(group))
                    end_day_list.extend([day_name] * len(group))
                    start_time_list.extend([start_time] * len(group))
                    end_time_list.extend([end_time] * len(group))
                    discount_factor_list.extend([1.2] * len(group))
            else:  # Weekends
                start_day_list.extend([day_name] * len(group))
                end_day_list.extend([day_name] * len(group))
                start_time_list.extend([weekend_time_range[0]] * len(group))
                end_time_list.extend([weekend_time_range[1]] * len(group))
                discount_factor_list.extend([weekend_discount_factor] * len(group))

    # Add new columns to the input DataFrame
    toll_rate_df['start_day'] = start_day_list[:len(toll_rate_df)]
    toll_rate_df['start_time'] = start_time_list[:len(toll_rate_df)]
    toll_rate_df['end_day'] = end_day_list[:len(toll_rate_df)]
    toll_rate_df['end_time'] = end_time_list[:len(toll_rate_df)]
    toll_rate_df['discount_factor'] = discount_factor_list[:len(toll_rate_df)]

    return toll_rate_df

# Example usage
file_path = "C://Users//mahes//MapUp-Data-Assessment-F//datasets//dataset-3.csv"
distance_matrix = calculate_distance_matrix(file_path)  # Assuming you have a function that calculates distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)   # Assuming you have a function that unrolls the distance matrix
toll_rate_df = calculate_toll_rate(unrolled_df)        # Assuming you have a function that calculates toll rates
time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_df)


# In[ ]:




