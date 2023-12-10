import pandas as pd

dataset1 = '../datasets/dataset-1.csv'
dataset2 = '../datasets/dataset-2.csv'

# task 1
# question 1
def generate_car_matrix(dataset):
    df = pd.read_csv(dataset)
    id2 = pd.Series(list(set(df.id_2.sort_values())))
    q1 = pd.DataFrame(0.0, index=pd.Index(id2, name='id_1'), columns=id2)
    q1 = q1.rename_axis(columns='id_2')
    for i in range(len(id2)):
        for j in range(len(id2)):
            car = df.loc[(df['id_1'] == id2[i]) & (df['id_2'] == id2[j]), 'car']
            if len(car) != 0:
                val = car.values[0]
            else:
                val = 0.0
            q1.at[id2[i], id2[j]] = val
    return q1

# question 2
def get_type_count(dataset):
    df = pd.read_csv(dataset)
    df['car_type'] = pd.cut(df['car'], bins=[0, 15, 25, float('inf')], labels=['low', 'medium', 'high'])
    return df

# question 3
def get_bus_indexes(dataset):
    df = pd.read_csv(dataset)
    mean_bus_value = df['bus'].mean()
    indices = df[df['bus'] > 2 * mean_bus_value].index
    return sorted(indices)

# question 4
def filter_routes(dataset):
    df = pd.read_csv(dataset)
    mean_truck_value = df['truck'].mean()
    routes = df[df['truck'] > 7]['route'].unique()
    return sorted(routes)

# question 5
def multiply_matrix(matrix):
    modified_df = matrix.copy()
    modified_df[modified_df > 20] *= 0.75
    modified_df[modified_df <= 20] *= 0.75
    modified_df = modified_df.round(1)
    return modified_df

# question 6

def checkValidity(df):
    day_mapping = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                   'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}

    # Replace values in the DataFrame
    df['startDay'] = df['startDay'].replace(day_mapping)
    df['endDay'] = df['endDay'].replace(day_mapping)
    df = df.sort_values(by = 'startDay')

    # print(df[['id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime']])
    startDay = None
    endDay = None
    startTime = None
    endTime = None

    for index, row in df.iterrows():
        # print(f"Index: {index}, Data: {row}")

        if(startDay == None):
            startDay = row.startDay
            startTime = row.startTime

        startDay = min(startDay, row.startDay)
        startTime = min(startTime, row.startTime)

        if endDay is None:
            endDay = row.endDay
            endTime = row.endTime
        elif endDay == row.endDay:
            endTime = max(endTime,row.endTime)
        elif endDay == row.startDay and endTime > row.startTime :
            endDay = max(endDay,row.endDay)
            endTime = max(endTime,row.endTime)

        elif ((row.startDay < endDay and row.endDay > endDay) or (row.startDay == endDay+1 and endTime=='23:59:59' and row.startTime =='00:00:00')):
            # endDay = max(endDay , row.endDay)
            if( endDay == row.endDay):
                endTime = max(endTime,row.endTime)
            else :
                endDay = row.endDay
                endTime = row.endTime

    if(startDay == 1 and endDay == 7 and startTime == '00:00:00' and endTime == "23:59:59"):
        return True
    else:
        return False








def time_check(dataset):
    df = pd.read_csv(dataset)

    df = df[['id','id_2','startDay', 'startTime', 'endDay', 'endTime']]
    # print(df[['id','id_2','startDay', 'startTime', 'endDay', 'endTime']])

    df = df.drop_duplicates()

    # check if 24hr completes 12:00:00 to 11:59:59
    days ={'Monday':1,'Tuesday':2,'Wednesday':3,
            'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}


    result = df.groupby(['id','id_2']).apply(checkValidity)

    multi_index_boolean_series = pd.Series(result,name='timestamp' ,index=result.index)

    return multi_index_boolean_series





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # df = pd.read_csv(dataset1)
    # print(df.tail(20))
    print(generate_car_matrix(dataset1))
    # print(get_type_count(dataset1))
    # print(get_bus_indexes(dataset1))
    # print(filter_routes(dataset1))
    # print(multiply_matrix(generate_car_matrix(dataset1)))
    # print(time_check(dataset2))
    pass
