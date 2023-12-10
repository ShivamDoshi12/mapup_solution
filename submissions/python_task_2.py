import pandas as pd
import numpy as np
from datetime import time

dataset1 = '../datasets/dataset-1.csv'
dataset3 = '../datasets/dataset-3.csv'

def calculate_distance_matrix(dataset):

    df = pd.read_csv(dataset)
    pivot_table = df.pivot_table(index='id_start', columns='id_end', values='distance',aggfunc='sum')

    filled_matrix = pivot_table.interpolate(method='linear', axis=1, limit_direction='both')

    id = df.id_start
    q1 = pd.DataFrame(0.0, index=pd.Index(id, name='id_end'), columns=id)
    q1 = q1.rename_axis(columns='id_start')


    for i in range(q1.shape[0]):
        for j in range(i):
            val = filled_matrix[id[i]][id[j]]
            # print(val)
            q1.at[id[i],id[j]] = val
            q1.at[id[j], id[i]] = val

    return q1


#question 2
def unroll_distance_matrix(df):
    id_starts = []
    id_ends = []
    distances = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            id_starts.append(df.index[i])
            id_ends.append(df.columns[j])
            distances.append(df.iloc[i, j])

    unfurled_df = pd.DataFrame({'id_start': id_starts,
                                'id_end': id_ends,
                                'distance': distances})

    return unfurled_df

#question 3
def find_ids_within_ten_percentage_threshold(df,reference_id):
    # Filter to rows with reference ID as id_start

    ref_rows = df[df['id_start'] == reference_id]

    # Calculate average distance for reference ID
    ref_avg_dist = ref_rows['distance'].mean()

    # Get 10% threshold
    ten_pct = 0.1 * ref_avg_dist
    lower = ref_avg_dist - ten_pct
    upper = ref_avg_dist + ten_pct

    # Filter rows within threshold
    close_rows = df[(df['distance'] >= lower) &
                    (df['distance'] <= upper)]

    # Return sorted list of id_starts
    return sorted(close_rows['id_start'].unique())

#question 4
def calculate_toll_rate(df):
    # Toll rate coefficients
    rates = {'moto': 0.8,
             'car': 1.2,
             'rv': 1.5,
             'bus': 2.2,
             'truck': 3.6}

    # Calculate toll rates for each vehicle type
    toll_cols = []
    for vehicle, rate in rates.items():
        toll = df['distance'] * rate
        toll_col = f'{vehicle}_toll'
        df[toll_col] = toll
        toll_cols.append(toll_col)

    return df


def calculate_time_based_toll_rates(df):
    num_hours = 24
    num_days = 7

    start_times = [time(hour=h) for h in range(num_hours)]
    # Fix end times to max 23
    end_times = [time(hour=h if h < 23 else 0) for h in range(num_hours)]
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday']

    start_days = days * (num_hours // num_days)
    end_days = start_days[1:] + [start_days[0]]

    # Weekday discounts
    wkdy_discounts = [0.8 if h < 10 or h >= 18 else 1.2 for h in range(num_hours)]

    # Build timestamp df
    time_df = pd.DataFrame({'start_day': start_days,
                            'end_day': end_days,
                            'start_time': start_times,
                            'end_time': end_times})

    # Merge and apply discounts
    merged_df = pd.merge(df, time_df, how='cross')
    discounts = [wkdy_discounts[t.hour] if d not in ['Saturday', 'Sunday'] else wkend_discount
                 for d, t in zip(merged_df['start_day'], merged_df['start_time'])]

    # Apply discounts
    for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
        merged_df[vehicle] *= discounts

        # Iterate over each unique start/end pair
        for (start, end), sub_df in df.groupby(['id_start', 'id_end']):
            # Merge time_df with the subsection
            timed_sub_df = pd.merge(sub_df, time_df, how='inner')

            # Apply discounts
            # Modify subsample in-place
            timed_sub_df[v] *= discounts

            # Concatenate subsets back
        timed_df = pd.concat(subset_dfs, ignore_index=True)

        return timed_df

    return merged_df
if __name__ == '__main__':

    q1 = calculate_distance_matrix(dataset3)
    q2 = unroll_distance_matrix(q1)
    reference_id = 1001400
    q3 = find_ids_within_ten_percentage_threshold(q2,reference_id)
    # print(q3)
    q4 = calculate_toll_rate(q2)
    