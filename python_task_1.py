#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("C:\\Users\\mahes\\MapUp-Data-Assessment-F\\datasets\\dataset-1.csv")

df.head()


# In[3]:


print(type(df))


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


df


# In[8]:


import pandas as pd

def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Pivot the DataFrame to create a matrix
   
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    
    car_matrix.values[[range(len(car_matrix))]*2] = 0

    
    return car_matrix


# In[9]:


car_matrix


# In[ ]:





# In[10]:


###2


# In[11]:


import pandas as pd

def get_type_count(df: pd.DataFrame) -> dict:
    """
    Adds a new categorical column 'car_type' based on values of the column 'car'.
    Calculates the count of occurrences for each 'car_type' category and returns the result as a dictionary.
    Sorts the dictionary alphabetically based on keys.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        dict: Count of occurrences for each 'car_type' category.
    """
    # Add a new categorical column 'car_type'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Example usage:
file_path = "C:\\Users\\mahes\\MapUp-Data-Assessment-F\\datasets\\dataset-1.csv"
df = pd.read_csv(file_path)
result = get_type_count(df)

# Display the result
print(result)


# In[ ]:





# In[12]:


##3


# In[13]:


import pandas as pd

def get_bus_indexes(df: pd.DataFrame) -> list:
    """
    Identify and return the indices where the 'bus' values are greater than twice the mean value of the 'bus' column.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        list: Indices where 'bus' values are greater than twice the mean value.
    """
    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage:
file_path = "C:\\Users\\mahes\\MapUp-Data-Assessment-F\\datasets\\dataset-1.csv"
df = pd.read_csv(file_path)
result = get_bus_indexes(df)

# Display the result
print(result)


# In[ ]:





# In[14]:


##4


# In[15]:


import pandas as pd

def filter_routes(df: pd.DataFrame) -> list:
    """
    Return the sorted list of values in the 'route' column for which the average of values in the 'truck' column is greater than 7.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        list: Sorted list of values in the 'route' column.
    """
    # Group by 'route' and calculate the average of 'truck' values
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

# Example usage:
file_path = "C:\\Users\\mahes\\MapUp-Data-Assessment-F\\datasets\\dataset-1.csv"
df = pd.read_csv(file_path)
result = filter_routes(df)

# Display the result
print(result)


# In[ ]:





# In[16]:


##5


# In[17]:


import pandas as pd

def multiply_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modify values in the DataFrame based on the specified logic:
    - If a value is greater than 20, multiply by 0.75.
    - If a value is 20 or less, multiply by 1.25.
    Round the modified values to 1 decimal place.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified DataFrame.
    """
    # Copy the DataFrame to avoid modifying the original DataFrame
    modified_df = df.copy()

    # Apply the specified logic to modify values
    modified_df = modified_df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the modified values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

# Example usage:
# Assuming df is the DataFrame obtained from Question 1
df = pd.read_csv("C:\\Users\\mahes\\MapUp-Data-Assessment-F\\datasets\\dataset-1.csv")
modified_result = multiply_matrix(df)

# Display the modified result
print(modified_result)


# In[ ]:





# In[18]:


##6


# In[19]:


df2=pd.read_csv("C:\\Users\\mahes\\MapUp-Data-Assessment-F\\datasets\\dataset-2.csv")
df2.head()


# In[ ]:





# In[20]:


import pandas as pd

def verify_timestamps(df: pd.DataFrame, start_day_column: str = 'startDay', start_time_column: str = 'startTime') -> pd.Series:
    """
    Verify the completeness of time data for each (id, id_2) pair.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns id, id_2, startDay, and startTime.
        start_day_column (str): Name of the startDay column.
        start_time_column (str): Name of the startTime column.

    Returns:
        pd.Series: Boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    # Combine 'startDay' and 'startTime' to create a 'timestamp' column
    df['timestamp'] = pd.to_datetime(df[start_day_column] + ' ' + df[start_time_column], errors='coerce', format='%Y-%m-%d %H:%M:%S')

    # Extract day and time components from timestamp
    df['day'] = df['timestamp'].dt.day_name()
    df['time'] = df['timestamp'].dt.time

    # Check if each (id, id_2) pair covers a full 24-hour period and spans all 7 days
    verification_result = df.groupby(['id', 'id_2']).apply(verify_pair).reset_index(level=['id', 'id_2'], drop=True)

    return verification_result

def verify_pair(group):
    """
    Helper function to verify a single (id, id_2) pair.

    Args:
        group (pandas.DataFrame): Subset of the DataFrame for a specific (id, id_2) pair.

    Returns:
        bool: True if the pair has incorrect timestamps, False otherwise.
    """
    # Check if the pair covers a full 24-hour period
    full_24_hours = len(group['time'].unique()) == 24

    # Check if the pair spans all 7 days
    spans_all_days = len(group['day'].unique()) == 7

    # Return True if either condition is not met
    return not (full_24_hours and spans_all_days)

# Example usage:
# Assuming df is the DataFrame obtained from dataset-2.csv
df = pd.read_csv("C:\\Users\\mahes\\MapUp-Data-Assessment-F\\datasets\\dataset-2.csv")
verification_result = verify_timestamps(df, start_day_column='startDay', start_time_column='startTime')

# Display the verification result
print(verification_result)


# In[ ]:




