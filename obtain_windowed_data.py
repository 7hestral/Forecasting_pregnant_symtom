from data.load_data import data_load
import pandas as pd
# from src.s3_utils import pandas_from_csv_s3
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from hyperimpute.plugins.imputers import Imputers
import os


SYMPTOM = 'edema'
activity_features = [
    # 'cal_active',
    # 'cal_total',
    # 'daily_movement',
    'high',
    'inactive',
    'low',
    'medium',
    'non_wear',
    'rest',
    'steps']
sleep_features = ['breath_average',
    'hr_average', 'rmssd', 'score', 'restless', 'hr_lowest']
def remove_outliers(df, feats):
    for column in feats:
        lower_quantile = df[column].quantile(0.01)
        upper_quantile = df[column].quantile(0.99)
        df.loc[(df[column] < lower_quantile) | (df[column] > upper_quantile), column] = np.nan
    return df



def get_usable_window(mask, window_size=28, tolerance=2):
    result_windows = []
    T = len(mask)
    i=0
    while i < (T - window_size + 1):
        curr_window = np.array(mask[i:i+window_size])
        # usable
        if np.sum(curr_window) < tolerance:
            result_windows.append((i, i+window_size))
            i += window_size
        else:
            # add tolerance window
            tolerance_window = np.array(mask[i:i+window_size+tolerance])
            miss_idx = np.where(tolerance_window == 1)[0]
            tolerance_counter = 0
            early_stop = False
            stop_idx = None
            for j in range(len(miss_idx) - 1):
                if miss_idx[j] + 1 == miss_idx[j + 1]:
                    tolerance_counter += 1
                else:
                    tolerance_counter = 0
                if tolerance_counter >= tolerance:
                    early_stop = True
                    stop_idx = j + 1
                    # break
            if early_stop:
                i += stop_idx + 1
            else:
                # usable
                result_windows.append((i, i+window_size))
                i += window_size
    return result_windows

def generate_csv_for_user(data, selected_user, preset_start_date=datetime(2009, 10, 12, 10, 10), preset_end_date=datetime(2030, 10, 12, 10, 10), file_name=''):
    root_path = os.path.join('/storage/home/ruizhu', 'dataset')
    # selected_user = 1441
    print(f"Curr user: {selected_user}")
    root_folder = file_name


    data_bodyport = data['bodyport']
    data_oura_activity = data['oura_activity']
    data_oura_sleep = data['oura_sleep']
    # data_survey = data['surveys']
    # survey_question_str = 'swollen'

    # selected_data_bodyport = data_bodyport[data_bodyport['user_id'] == selected_user][['date', 
    # # 'impedance_ratio', 
    # # 'peripheral_fluid', 
    # 'impedance_mag_1_ohms', 'impedance_phase_1_degs', 
    # 'weight_kg']].groupby("date", as_index = False).first()

    # the formatted files were  generated from relabel_edema.ipynb
    # survey_question_str_lst = ['mood', 'fatigue']
    survey_question_str_lst = [SYMPTOM]
    # edema_csv_path = f'/mnt/results/edema_coarse_label/user_{selected_user}_edema_coarse_label.csv'
    # if not os.path.exists(edema_csv_path):
    #     print("Empty Edema")
    #     return
    selected_data_survey_lst = []
    for q_str in survey_question_str_lst:
        csv_path = os.path.join(root_path, f'{q_str}/user_{selected_user}_{q_str}_label.csv')
        if not os.path.exists(csv_path):
            print(f"Empty {q_str}")
            return
        survey_question_csv = pd.read_csv(csv_path).groupby("date", as_index = False).first()
        survey_question_csv = survey_question_csv.fillna(value=1.0)
        
        selected_data_survey_lst.append(survey_question_csv)


    selected_data_oura_sleep = data_oura_sleep[data_oura_sleep['user_id'] == selected_user][sleep_features+['date']]
    selected_data_oura_activity = data_oura_activity[data_oura_activity['user_id'] == selected_user][activity_features+['date']]


    def get_min_date(df):
        return np.min(df['date'].astype('datetime64[ns]'))
    def get_max_date(df):
        return np.max(df['date'].astype('datetime64[ns]'))
    
    # if not len(selected_data_bodyport):
    #     print("Empty bodyport")
    #     return
    if not len(selected_data_oura_activity):
        print("Empty Oura activity")
        return
    # if not len(selected_data_edema):
    #     print("Empty Edema")
    #     return
    if not len(selected_data_oura_sleep):
        print("Empty Oura sleep")
        return
    ds_lst = [# selected_data_edema, #selected_data_bodyport, 
    selected_data_oura_activity, selected_data_oura_sleep] + selected_data_survey_lst
    overall_min_date = np.max(list(map(get_min_date, ds_lst)) + [preset_start_date])
    overall_max_date = np.min(list(map(get_max_date, ds_lst)) + [preset_end_date])
    
    date_range = pd.date_range(overall_min_date, overall_max_date, freq='d')
    print(overall_max_date-overall_min_date)
    if overall_max_date-overall_min_date < timedelta(days=10):
        return False
    date_df = pd.DataFrame()
    date_df['date'] = date_range
    date_df['date'] = date_df['date'].astype("datetime64[ns]")
    date_df.to_csv('~/test2.csv')
    def change_date_type(df):
        df['date'] = df['date'].astype('datetime64[ns]')
        return pd.merge(date_df, df, how='left')

    # selected_data_edema = change_date_type(selected_data_edema)
    for i in range(len(ds_lst)):
        ds_lst[i] = change_date_type(ds_lst[i])
    # selected_data_oura_activity = change_date_type(selected_data_oura_activity)
    # selected_data_oura_sleep = change_date_type(selected_data_oura_sleep)
    # selected_data_bodyport = change_date_type(selected_data_bodyport)

    unimputed_df = pd.DataFrame()
    unimputed_df['date'] = date_range
    # unimputed_df = pd.merge(unimputed_df, selected_data_bodyport, how='left')
    # unimputed_df = pd.merge(unimputed_df, selected_data_oura_activity, how='left')
    # unimputed_df = pd.merge(unimputed_df, selected_data_oura_sleep, how='left')
    # unimputed_df = pd.merge(unimputed_df, selected_data_edema, how='left')
    for i in range(len(ds_lst)):
        unimputed_df = pd.merge(unimputed_df, ds_lst[i], how='left')

    missingness_mask = unimputed_df.isna()
    missingness_df = missingness_mask.copy().astype(int)
    missingness_mask = np.sum(missingness_mask, axis=1)
    missingness_mask[missingness_mask >= 1] = 1
    
    window_lst = get_usable_window(missingness_mask, tolerance=3)

    imputers = Imputers()
    imputers.list()
    method = 'hyperimpute'
    plugin = Imputers().get(method)


    root_path = os.path.join('/storage/home/ruizhu', 'dataset')
    if not os.path.exists(os.path.join(root_path, root_folder)):
        os.mkdir(os.path.join(root_path, root_folder))
    # save the windows
    train_length = 14
    for count, w in enumerate(window_lst):
        curr_window = unimputed_df[w[0]:w[1]]
        X = curr_window.drop('date', axis=1).drop('user_id', axis=1)
        X_missingness = missingness_df[w[0]:w[1]]
        X_copy = X.copy()
        
        X_copy.reset_index(inplace=True, drop=True)
        X_train = X_copy[:train_length]
        X_test = X_copy[train_length:]
        X_train = plugin.fit_transform(X_train)
        X_test = plugin.transform(X_test)
        X_copy[:train_length] = X_train
        X_copy[train_length:] = X_test
        # curr_window = plugin.fit_transform(X_copy)
        X_copy.to_csv(os.path.join(root_path, f'{root_folder}/user_{selected_user}_{file_name}_hyperimpute_slice{count}.csv'), index=False, header=False)
        X_missingness.to_csv(os.path.join(root_path, f'{root_folder}/user_{selected_user}_{file_name}_missingness_slice{count}.csv'), index=False)

        hyperimputed_df_with_date = X_copy.copy()
        hyperimputed_df_with_date['date'] = date_range[w[0]:w[1]]
        hyperimputed_df_with_date.to_csv(os.path.join(root_path, f'{root_folder}/user_{selected_user}_{file_name}_hyperimpute_with_date_slice{count}.csv'), index=False)

    return len(window_lst)


if __name__ == "__main__":
    data = data_load(data_keys={'bodyport', 'oura_activity', 'oura_sleep', "surveys"})
    print(np.sum(data['oura_activity'].isna()))
    print('max', data['oura_activity']['steps'].max())
    print('min', data['oura_activity']['steps'].min())
    print('0.95', data['oura_activity']['steps'].quantile(0.99))
    print('0.05', data['oura_activity']['steps'].quantile(0.01))
    data['oura_sleep'] = remove_outliers(data['oura_sleep'], sleep_features)
    data['oura_activity'] = remove_outliers(data['oura_activity'], activity_features)
    print(np.sum(data['oura_activity'].isna()))
    print('max', data['oura_activity']['steps'].max())
    print('min', data['oura_activity']['steps'].min())
    print('0.95', data['oura_activity']['steps'].quantile(0.99))
    print('0.05', data['oura_activity']['steps'].quantile(0.01))
    # exit(0)
    
    df_birth = data_load(data_keys={"birth"})['birth']

    counter = 0
    available_user = []
    if SYMPTOM == 'fatigue':
        reliable_user = [39, 84, 122, 137, 142, 174, 214, 225, 431, 581, 962, 966, 1002, 1032, 1363, 1373, 1387, 1437, 1440, 1697, 1715, 1719, 1730, 1744, 1753, 2015, 2018, 2058, 2062, 2068, 2109, 2150, 2153, 2159, 2167, 2214, 2265, 2312, 2318, 2339, 2340, 2347, 2370, 2386, 2482, 2484, 2488, 2500, 2514, 2516, 2572, 2578, 2583, 2598, 2603, 2607, 2610, 2612, 2664]
    elif SYMPTOM == 'edema':
        reliable_user = [28, 29, 30, 35, 37, 38, 39, 42, 44, 45, 47, 53, 54, 64, 65, 66, 67, 68, 74, 75, 79, 80, 94, 95, 1431, 1393, 99, 114, 118, 119, 2093, 122, 124, 127, 135, 137, 155, 1014, 156, 158, 1373, 168, 1000, 173, 174, 185, 186, 1001, 190, 192, 193, 199, 200, 212, 1021, 972, 1724, 1004, 1429, 234, 975, 280, 289, 290, 291, 292, 404, 405, 407, 408, 409, 410, 977, 1047, 428, 581, 594, 595, 600, 603, 615, 1658, 1045, 983, 966, 969, 985, 987, 989, 991, 992, 1005, 1455, 1038, 1044, 2117, 2225, 2226, 2120, 1363, 1366, 1367, 1378, 1719, 2151, 1383, 1386, 1387, 1721, 1425, 1426, 1427, 1432, 1436, 1439, 1440, 1443, 1697, 1452, 1700, 1703, 1706, 1707, 1708, 1709, 1710, 1712, 1715, 1716, 1728, 1731, 2299, 1743, 1745, 1749, 1750, 2187, 2091, 2197, 1976, 1988, 1989, 1994, 1995, 1996, 1997, 1999, 2000, 2001, 2004, 2014, 2016, 2018, 2019, 2038, 2032, 2056, 2058, 2060, 2061, 2062, 2064, 2068, 2076, 2202, 2645, 2654, 2084, 2223, 2311, 2100, 2102, 2109, 2158, 2159, 2160, 2166, 2169, 2174, 2176, 2178, 2212, 2235, 2664, 2650, 2265, 2339, 2340, 2341, 2342, 2260, 2510, 2615, 2147, 2145, 2139, 2133, 2164, 2182, 2183, 2167, 2201, 2204, 2203, 2261, 2259, 2257, 2313, 2312, 2351, 2350, 2379, 2386, 2381, 2516, 2541, 2518, 2530, 2487, 2490, 2496, 2489, 2485, 2500, 2483, 2502, 2503, 2547, 2592, 2593, 2549, 2580, 2581, 2536, 2599, 2602, 2572, 2603, 2571, 2583, 2574, 2584, 2575, 2628, 2576, 2611, 2562, 2631, 2613, 2635, 2636, 2564, 2565, 2551, 2637, 2656]                 
    for user in reliable_user:
        user = int(user)
        if len(df_birth[df_birth.user_id == user].birth_date):
            curr_birth_date = pd.to_datetime(df_birth[df_birth.user_id == user].birth_date.values[0])
            # only interested in the 3rd trimester
            
            # only interested in the 1st trimester
            interested_start_date = curr_birth_date - timedelta(days=1*91)
            interested_end_date = curr_birth_date - timedelta(days=0*91)
            result = generate_csv_for_user(data, user, preset_start_date=interested_start_date, preset_end_date=interested_end_date, file_name=f'{SYMPTOM}_analysis_3rd_trend')
            if result:
                available_user.append(user)
                counter += result

    # user_dict = {}

    # for f in os.listdir(os.path.join("/", "mnt", 'results', "edema_pred_window")):

    #     f_name_lst = f.split('_')
    #     if 'date' in f_name_lst:
    #         continue
        
    #     user_id = int(f_name_lst[1])
    #     if user_id in user_dict:
    #         user_dict[user_id] += 1
    #     else:
    #         user_dict[user_id] = 1
    # print(user_dict)
    # s = []
    # for i in user_dict:
    #     s.append(user_dict[i])
    # sns.histplot(s)
    