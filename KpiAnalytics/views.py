import json
from django.utils.safestring import mark_safe
from django.shortcuts import render
from django.http import HttpResponse
import io
import pandas as pd
from collections import defaultdict
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
import os
import shutil
from openai import OpenAI
import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.impute import KNNImputer
from django.core.cache import cache
from django.shortcuts import redirect
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pmdarima as pm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import xml.etree.ElementTree as ET
from keras.models import load_model
import xmltodict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Configure OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)


def testing(request):
    return HttpResponse("Hurrah")


KPI_LOGICS = defaultdict()
checks = []

if os.path.exists('uploads'):
    shutil.rmtree('uploads')
os.makedirs('uploads', exist_ok=True)

if os.path.exists('kpis.json'):
    os.remove('kpis.json')


def updatedtypes(df):
    datatypes = df.dtypes
    for col in df.columns:
        if datatypes[col] == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                pass
    return df


def updateddtypes(df):
    datatypes = df.dtypes
    for col in df.columns:
        if datatypes[col] == 'object':
            try:
                pd.to_datetime(df[col])
                df.drop(col, axis=1, inplace=True)
                print(df.columns)
            except Exception as e:
                pass
    return df


def get_importance(X_train, y_train, model_type):
    if model_type == 'regression':
        model_ = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model_ = RandomForestClassifier(n_estimators=100, random_state=42)
    model_.fit(X_train, y_train)

    # Get feature importances
    feature_importance = model_.feature_importances_

    # Normalize feature importance to percentages
    feature_importance_percent = (feature_importance / np.sum(feature_importance)) * 100

    # Print feature importance scores in percentage
    df = pd.DataFrame({"Features": X_train.columns, "Importances": feature_importance_percent})
    df.sort_values(by='Importances', inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)
    return df


def iscatcol(col, t, threshold=10):
    unique_values = col.dropna().unique()
    if len(unique_values) <= threshold or t == 'object':
        if t == 'object':
            return True, True  # Categorical and needs encoding
        return True, False  # Categorical but doesn't require encoding
    return False, False


def getcatcols(df):
    catcols = []
    catcols_encode = []
    unique_cols = {}
    for col in df.columns:
        a, b = iscatcol(df[col], df.dtypes[col])
        if a:
            catcols.append(col)
        if b:
            catcols_encode.append(col)
            unique_cols[col] = list(df[col].unique())
    return catcols, catcols_encode, unique_cols


def convert_to_datetime(df):
    # Define the possible date formats to try
    date_formats = ['%m-%d-%Y', '%m/%d/%Y', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d']

    # Loop through each column
    for col in df.columns:
        # Only process object columns, assuming they may contain dates in string format
        if df[col].dtype == 'object':
            # Check if the column contains potential date strings
            if df[col].str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', na=False).any():
                # Try to parse automatically first
                try:
                    df[col] = pd.to_datetime(df[col], errors='raise')
                except (ValueError, TypeError):
                    # If automatic parsing fails, try each format individually
                    def parse_date(value):
                        for fmt in date_formats:
                            try:
                                return pd.to_datetime(value, format=fmt)
                            except (ValueError, TypeError):
                                continue
                        return pd.NaT  # Return NaT if none of the formats match

                    # Apply the custom parse function to handle multiple formats
                    df[col] = df[col].apply(parse_date)
    return df


def home_page(request):
    try:
        if request.method == "GET":
            if os.path.exists('uploads'):
                if os.path.exists(os.path.join('uploads', 'data.csv')):
                    uploadedInfo = "True"
                    data_frame = pd.read_csv(os.path.join('uploads', 'data.csv'))
                    data_frame = updatedtypes(data_frame)
                    with open('uploads/configs.json', 'r') as json_file:
                        data = json.load(json_file)
                        return render(request, 'dashboard.html',
                                      {
                                          "uploadedInfo": "True",
                                          'df': data_frame,
                                          "data_file_name": data['data_file_name'],
                                          "kpi_config_file_name": data["kpi_config_file_name"]
                                      }
                                      )
            return render(request, 'dashboard.html')
        else:
            files = request.FILES['file']
            kpi_file = request.FILES.get("kpi_file")
            if len(files) < 1:
                return HttpResponse('No files uploaded')
            else:
                print('jakass')
                for item in os.listdir('models'):
                    print('hh')
                    item_path = os.path.join('models', item)
                    print(item_path)
                    # Check if the item is a directory
                    if os.path.isdir(item_path):
                        # Remove the directory and all its contents

                        shutil.rmtree(item_path)
                data_file_name, kpi_config_file_name = '', ''
                content = files.read().decode('utf-8')
                csv_data = io.StringIO(content)
                df = pd.read_csv(csv_data)
                df = updatedtypes(df)
                df.to_csv(os.path.join("uploads", 'data.csv'), index=False)
                data_file_name = files.name
                new_df, html_df = process_missing_data(df.copy())
                cache.set('dataframe', html_df)
                request.session['dataframe'] = html_df
                new_df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)
                if os.path.exists('kpis.json'):
                    os.remove('kpis.json')
                request.session['uploadedFileName'] = files.name
                if kpi_file:
                    kpis_dict = xmltodict.parse(kpi_file.read())
                    with open('uploads/kpi_config.json', 'w') as json_file:
                        json.dump(kpis_dict, json_file, indent=4)
                        kpi_config_file_name = kpi_file.name
                with open('uploads/configs.json', 'w') as json_file:
                    json.dump({
                        "data_file_name": data_file_name,
                        "kpi_config_file_name": kpi_config_file_name}, json_file, indent=4)
                return render(request, 'dashboard.html',
                              {
                                  "uploadedInfo": "True",
                                  "df": df,
                                  "data_file_name": data_file_name,
                                  "kpi_config_file_name": kpi_config_file_name
                              }
                              )
    except Exception as e:
        print(e)


def data_process(request):
    if request.method == 'GET':
        if os.path.exists(os.path.join('uploads', 'processed_data.csv')):
            df = pd.read_csv(os.path.join('uploads', 'processed_data.csv'))
            df = updatedtypes(df)
            if df.shape[0] > 0:
                nullvalues = df.isnull().sum().to_dict()
                parameters = list(nullvalues.keys())
                Count = list(nullvalues.values())
                total_missing = df.isnull().sum().sum()
                # df, html_df = process_missing_data(df)
                # cache.set('dataframe', html_df)
                # df.to_csv(os.path.join('uploads', 'processed_data.csv'), index=False)
                nor = df.shape[0]
                nof = df.shape[1]
                timestamp = 'N'
                boolean = 'N'
                categorical_vars = []
                boolean_vars = []
                numeric_vars = {}
                datetime_vars = []
                text_data = []
                td = None
                stationary = "NA"
                numfilter = ['25%', '50%', '75%']
                single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
                df.drop(single_value_columns, axis=1, inplace=True)
                for i, j in df.dtypes.items():
                    if str(j) in ["float64", "int64"]:
                        data = df[i].describe().to_dict()
                        temp = [data.pop(key) for key in numfilter]
                        numeric_vars[i] = data
                    elif str(j) in ["object"] and i not in ['Remark']:
                        categorical_vars.append({i: df[i].nunique()})
                    elif str(j) in ["datetime64[ns]"]:
                        if i.upper() in ['DATE', "TIME", "DATE_TIME"]:
                            td = i
                        datetime_vars.append(i)
                    elif str(j) in ["bool"]:
                        boolean_vars.append(i)
                print("A")
                request.session['TimeSeriesColumns'] = datetime_vars
                if 'Remark' in df.columns:
                    text_data.append('Remark')
                print("B")
                istextdata = 'Y' if len(text_data) > 0 else 'N'
                if len(datetime_vars) > 0:
                    timestamp = 'Y'
                if td:
                    stationary = adf_test(df, td)
                print("C")
                catvalues = [{'Parameter': list(data.keys())[0], 'Count': list(data.values())[0]} for data in
                             categorical_vars]
                sentiment = checkSentiment(df, categorical_vars)
                print("D")
                if len(catvalues) > 0:
                    catdf = pd.DataFrame(catvalues).to_html(classes='mystyle')
                else:
                    catdf = 'NA'
                if len(numeric_vars) > 0:
                    numdf = pd.DataFrame(numeric_vars).T
                    numdf.columns = ['Count', 'Mean', 'Std', 'Min', 'Max']
                    numdf = numdf.to_html(classes='mystyle')
                else:
                    numdf = 'NA'
                print("E")
                if len(boolean_vars) > 0:
                    boolean = 'Y'

                missingvalue = pd.DataFrame({"Parameters": parameters, 'Missing Value Count': Count})

                duplicate_records = df[df.duplicated(keep='first')].shape[0]
                print("F")
                for d in ['bar', 'pie', 'wordCloud']:
                    if os.path.exists(os.path.join(os.getcwd(), f'static/plots/{d}/')):
                        for f in os.listdir(os.path.join(os.getcwd(), f'static/plots/{d}/')):
                            os.remove(os.path.join(os.path.join(os.getcwd(), f'static/plots/{d}/'), f))
                    else:
                        os.makedirs(os.path.join(os.getcwd(), f'static/plots/{d}/'), exist_ok=True)
                plot_numeric(numeric_vars, df)

                plot_categorical(categorical_vars, df)
                plot_wordCloud(text_data, df)
                barplots = {i.split('_')[1].split('.')[0]: i for i in
                            os.listdir(os.path.join(os.getcwd(), 'static/plots/bar/')) if
                            i.endswith('png') and i.split('_')[0] == 'Bar'}

                pieplots = {i.split('_')[1].split('.')[0]: i for i in
                            os.listdir(os.path.join(os.getcwd(), 'static/plots/pie/'))
                            if
                            i.endswith('png') and i.split('_')[0] == 'Pie'}

                wordColudPlots = {i.split('_')[1].split('.')[0]: i for i in
                                  os.listdir(os.path.join(os.getcwd(), 'static/plots/wordCloud/'))
                                  if
                                  i.endswith('png') and i.split('_')[0] == 'wordCloud'}

                return render(request, 'dataprocess.html',
                              {'nor': nor, 'nof': nof, 'timestamp': timestamp,
                               "single_value_columns": ",".join(single_value_columns) if len(
                                   single_value_columns) > 0 else "NA",
                               "sentiment": sentiment,
                               "stationary": stationary,
                               'catdf': catdf,
                               'missing_data': total_missing,
                               'numdf': numdf, 'boolean': boolean,
                               'missingvalue': missingvalue.to_html(classes='mystyle'),
                               'textdata': istextdata, 'duplicate_records': duplicate_records,
                               'batplots': barplots.items(),
                               'pieplots': pieplots.items(),
                               'wordCloudPlots': wordColudPlots.items(),
                               'form1': 'True'}
                              )
            else:
                return HttpResponse("No data")
        else:
            return render(request, 'dataprocess.html',
                          {'msg': 'Please upload file'})
    else:
        return HttpResponse('Invalid Request')


def get_prompt(request):
    try:
        if request.method == "POST":
            global KPI_LOGICS, checks
            KPI_LOGICS = defaultdict()
            checks = []
            prompt = request.POST.get('prompt')
            if not os.path.exists(os.path.join('uploads', 'data.csv')):
                return HttpResponse('No files uploaded')
            else:
                df = pd.read_csv(os.path.join('uploads', 'processed_data.csv'))
                df.to_csv('data.csv', index=False)
                prompt_desc = (
                    f"You are analytics_bot. Analyse the data: {df.head()} and for the uer query {prompt}, "
                    f"generate kpis with response as KPI Name, Column and Logic. Response should be in python dictionary format  with kpi names as keys. In response dont add any other information just provide only the response dictionary"
                )
                n = 2
                while n > 0:
                    res, kpis = generate_code(prompt_desc)
                    if res is not None:
                        if not os.path.exists('kpis.json'):
                            kpis_store = dict()
                        else:
                            with open('kpis.json', 'r') as fp:
                                kpis_store = json.load(fp)
                        with open('kpis.json', 'w') as fp:
                            kpis_store.update(kpis)
                            json.dump(kpis_store, fp)
                        break
                if os.path.exists(os.path.join('uploads', 'kpi_config.json')):
                    with open('uploads/kpi_config.json', 'r') as json_file:
                        kpis_dict = json.load(json_file)
                    for kpi in kpis_dict['Kpis']['kpi']:
                        kpis[kpi['KPI_Name']] = kpi
                        checks.append(kpi['KPI_Name'])
                return render(request, 'kpiprocess.html',
                              {
                                  'form1': 'True',
                                  'response': 'True', "kpis": kpis, "checks": checks}
                              )
        else:
            return render(request, 'kpiprocess.html',
                          {
                              'form1': 'True'}
                          )
    except Exception as e:
        print(e)


def convert_to_bool(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 2:
                df[col] = df[col].astype(bool)
    return df


def convert_to_category(df, threshold=10):
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= threshold:
            df[col] = df[col].astype('category')
    return df


def mvt(request):
    df_data = cache.get('dataframe', [])
    if df_data:
        df_data = request.session.get('dataframe')
        return render(request, 'MVT.html',
                      {
                          "df": df_data
                      }
                      )
    else:
        return render(request, 'MVT.html',
                      {
                          'msg': 'No data'
                      }
                      )


def handle_missing_data(df):
    try:
        # Identify numeric and datetime columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        date_time_cols = df.select_dtypes(include=['datetime64']).columns

        # Impute numeric columns and track which cells were imputed
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric = imputer.fit_transform(df[numeric_cols])
        imputed_numeric_df = pd.DataFrame(imputed_numeric, columns=numeric_cols)

        # Mark imputed cells (True if the original cell was NaN)
        imputed_flags = df[numeric_cols].isnull()
        imputed_flags = imputed_flags.applymap(lambda x: x if x else False)

        # Update DataFrame with imputed values
        df[numeric_cols] = imputed_numeric_df

        # Handle datetime columns by forward filling missing values
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])
            time_diffs = df[col].diff().dropna()
            avg_diff_sec = time_diffs.mean().total_seconds()
            minute_sec = 60
            hour_sec = 3600
            day_sec = 86400
            month_sec = day_sec * 30.44
            year_sec = day_sec * 365.25

            if avg_diff_sec < hour_sec:
                time_unit = "minutes"
                avg_diff = pd.Timedelta(minutes=avg_diff_sec / minute_sec)
            elif avg_diff_sec < day_sec:
                time_unit = "hours"
                avg_diff = pd.Timedelta(hours=avg_diff_sec / hour_sec)
            elif avg_diff_sec < month_sec:
                time_unit = "days"
                avg_diff = pd.Timedelta(days=avg_diff_sec / day_sec)
            elif avg_diff_sec < year_sec:
                time_unit = "months"
                avg_diff = pd.DateOffset(months=round(avg_diff_sec / month_sec))
            else:
                time_unit = "years"
                avg_diff = pd.DateOffset(years=round(avg_diff_sec / year_sec))

            for i in range(1, len(df)):
                if pd.isnull(df[col].iloc[i]):
                    df.loc[i, col] = df[col].iloc[i - 1] + avg_diff
                    imputed_flags.loc[i, col] = True

            imputed_flags.fillna(False, inplace=True)

        # Convert the DataFrame into a JSON-serializable format with flags
        data = []
        for _, row in df.iterrows():
            row_data = {}
            for col in df.columns:
                row_data[col] = {
                    "value": row[col].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row[col], pd.Timestamp) else row[col],
                    "is_imputed": str(imputed_flags[col].get(_, False)) if col in imputed_flags else str(False)
                    # Check if cell was imputed
                }
            data.append(row_data)
        return df, data
    except Exception as e:
        print(e)


def process_missing_data(df):
    df = convert_to_datetime(df)
    df, html_df = handle_missing_data(df)
    return df, html_df


@csrf_exempt
def generate_code(prompt_eng):
    try:
        global KPI_LOGICS
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_eng}
            ]
        )
        all_text = ""
        # Display generated content dynamically
        for choice in response.choices:
            message = choice.message
            chunk_message = message.content if message else ''
            all_text += chunk_message
        all_text = all_text.lower().replace('```python', '').replace('```', '')
        print(all_text)
        data_dict = json.loads(all_text)
        print("datadict", data_dict)
        for key, value in data_dict.items():
            if 'kpi name' in value:
                kpi_name = value['kpi name']
            elif 'name' in value:
                kpi_name = value["name"]
            else:
                kpi_name = key
            KPI_LOGICS[key] = {
                "KPI Name": kpi_name,
                "Column": value["column"],
                "Logic": value["logic"]
            }
        return all_text, KPI_LOGICS
    except Exception as e:
        print(e)
        return None, None


def extract_kpi(text):
    try:
        kpis = {}
        lines = text.strip().splitlines()
        current_kpi = None
        collecting_logic = False
        for line in lines[1:-1]:
            line = line.strip()

            if "KPI Name" in line:
                # Start a new KPI entry
                current_kpi = line.split(":", 1)[1].strip()
                kpis[current_kpi] = {}
                kpis[current_kpi]['KPI Name'] = current_kpi
                collecting_logic = False

            elif "Column" in line:
                if current_kpi:
                    kpis[current_kpi]['Column'] = line.split(":", 1)[1].strip()
                collecting_logic = False

            elif "Logic" in line:
                if current_kpi:
                    if len(line.split(":", 1)) > 0:
                        kpis[current_kpi]['Logic'] = line.split(":", 1)[1].strip()
                    collecting_logic = True  # Start collecting multi-line logic

            elif collecting_logic and current_kpi:
                kpis[current_kpi]['Logic'] += ' ' + line.strip()

        return kpis
    except Exception as e:
        print(e)


def kpi_code(request):
    try:
        if request.method == "POST":
            kpi_list = request.POST.getlist("kpi_names")
            paths, codes = generate_kpi_code(kpi_list)
            return render(request, 'kpiprocess.html',
                          {
                              'form1': 'True', 'plots': paths.items(),
                              'response': 'True', 'code': mark_safe(codes), "kpis": KPI_LOGICS, "checks": checks}
                          )
    except Exception as e:
        print(e)


def generate_kpi_code(kpi_list):
    try:
        df = pd.read_csv("data.csv")
        df = updatedtypes(df)
        codes = ''
        paths = {}
        for f in os.listdir(os.path.join(os.getcwd(), f'static/charts')):
            os.remove(os.path.join(os.path.join(os.getcwd(), f'static/charts/'), f))
        for kpi in kpi_list:
            prompt_desc = (
                f"You are analytics_bot. Read the data from data.csv file with example data as {df.head()} and generate python code with kpi details as {KPI_LOGICS[kpi]}. Save result in variable named result, plot a suitable plot for the result obtained, save it as name based on kpi and use static\charts to save the file."
                f"If length of result variable is 1 then keep bar width thin and x axis limit as -0.5 and 0.5"
            )
            code = ''
            try:
                code += generate_code2(prompt_desc)
            except Exception as e:
                print(e)
                code += f'Code generation failed for {kpi}'
            codes += "<b>" + kpi.capitalize() + "</b>" + "\n" + code + '\n'

        for path in os.listdir('static/charts'):
            paths[path[:-4]] = path
        return paths, codes
    except Exception as e:
        print(e)


@csrf_exempt
def generate_code2(prompt_eng):
    trials = 2
    try:
        while trials > 0:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_eng}
                ]
            )
            all_text = ""

            # Display generated content dynamically
            for choice in response.choices:
                print(f"Debug - choice structure: {choice}")  # Debugging line
                message = choice.message
                print(f"Debug - message structure: {message}")  # Debugging line
                chunk_message = message.content if message else ''
                all_text += chunk_message

            print(all_text)
            python_chuncks = all_text.count("```python")
            idx = 0
            code = ''
            for i in range(python_chuncks):
                code_start = all_text[idx:].find("```python") + 9
                code_end = all_text[idx:].find("```", code_start)
                code += all_text[idx:][code_start:code_end]
                idx = code_end
            print(code)
            try:
                local_vars = {}
                exec(code, {}, local_vars)
                code += f"\n <b>Output: {local_vars['result']}</b> \n <hr>"
                return code
            except Exception as e:
                print(e)
                trials -= 1
    except Exception as e:
        print(e)


def kpi_store(request):
    try:
        if request.method == "POST":
            kpi_list = request.POST.getlist("kpi_names")
            paths, codes = generate_kpi_code(kpi_list)
            return render(request, 'kpi_store.html',
                          {
                              'form1': 'True', 'plots': paths.items(),
                              'response': 'True', 'code': mark_safe(codes), "kpis": KPI_LOGICS, "checks": checks}
                          )
        else:
            with open('kpis.json', 'r') as fp:
                kpis_store = json.load(fp)
            return render(request, 'kpi_store.html',
                          {
                              'form1': 'True',
                              'response': 'True', "kpis": kpis_store, "checks": checks}
                          )
    except Exception as e:
        print(e)
        return render(request, 'kpi_store.html',
                      {
                          'form1': 'True',
                          'response': 'True', "kpis": {}}
                      )


@csrf_exempt
def models(request):
    try:
        if not os.path.exists(os.path.join("uploads", 'processed_data.csv')):
            return render(request, 'models.html',
                          {
                              "msg": 'Please upload file to continue'
                          }
                          )

        df = pd.read_csv(os.path.join("uploads", 'processed_data.csv'))
        rf_result = request.session.get('rf_result', '')
        if rf_result:
            request.session['rf_result'] = ''
            return render(request, 'models.html',
                          {
                              'form1': True,
                              'columns': df.columns,
                              'rf_result': rf_result
                          }
                          )
        print('hello')
        if request.method == 'POST':
            model_type = request.POST.get('model')
            col = request.POST.get('col')
            request.session['col_predict'] = col
            if model_type == 'RandomForest':
                stat, cols = random_forest(df, col)
                return render(request, 'models.html',
                              {
                                  'form1': True,
                                  'columns': df.columns,
                                  "rf": True,
                                  "status": stat,
                                  "rf_cols": cols
                              })
            elif model_type == "K-Means":
                stat, clustered_data = kmeans_train(df)
                return render(request, 'models.html',
                              {
                                  'form1': True,
                                  'columns': df.columns,
                                  "cluster": True,
                                  "status": stat,
                                  "clustered_data": clustered_data
                              })
            elif model_type == "Arima":
                stat = arima_train(df, col)
                path = f"../models/arima/{col}/actual_vs_forecast.png"
                return render(request, 'models.html',
                              {
                                  'form1': True,
                                  'columns': df.columns,
                                  "status": stat,
                                  "arima": True,
                                  "path": path,
                              })
            elif model_type == 'OutlierDetection':
                print('Hai')
                res = detect_outliers_zscore(df, col)

                return render(request, 'models.html',
                                  {
                                      'form1': True,
                                      'columns': df.columns,
                                      "status": True,
                                      "processed_data": res,
                                      "OutlierDetection": True
                                  })
        else:
            return render(request, 'models.html',
                          {
                              'form1': True,
                              'columns': df.columns}
                          )
    except Exception as e:
        print(e)
        return render(request, 'models.html',
                      {
                          'form1': False,
                          "msg": str(e)}
                      )


def outliercheck(df, column):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'detect outliers for  the following data {df[column]}'}
        ]
    )
    all_text = ""
    # Display generated content dynamically
    for choice in response.choices:
        message = choice.message
        chunk_message = message.content if message else ''
        all_text += chunk_message
    print(all_text)
    return all_text


def detect_outliers_zscore(df, column, threshold=3):
    try:

        res = outliercheck(df, column)

        # # Select numeric columns only
        # numeric_cols = df.select_dtypes(include=np.number)
        #
        # # Calculate Z-Scores for each numeric column
        # z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
        #
        # # Calculate an aggregate Z-Score for each row (e.g., max absolute Z-Score)
        # df['Row_Z-Score'] = z_scores.abs().max(axis=1)
        #
        # # Flag rows where the aggregate Z-Score exceeds the threshold
        # df['Outlier'] = df['Row_Z-Score'].apply(lambda x: 'Yes' if x > threshold else 'No')
        # df.drop('Row_Z-Score', axis=1,inplace=True)
        return res
    except Exception as e:
        print(e)


def find_elbow_point(inertia_values):
    # Calculate the rate of change between successive inertia values
    changes = np.diff(inertia_values)
    # Identify the elbow as the point where change starts to decrease
    elbow_point = np.argmin(np.abs(np.diff(changes))) + 1
    return elbow_point


def arima_train(data, target_col):
    try:
        # Identify date column by checking for datetime type
        date_column = None
        if not os.path.exists(os.path.join("models", 'arima', target_col)):
            os.makedirs(os.path.join("models", 'arima', target_col), exist_ok=True)
            for col in data.columns:
                if data.dtypes[col] == 'object':
                    try:
                        # Attempt to convert column to datetime
                        pd.to_datetime(data[col])
                        date_column = col
                        break
                    except (ValueError, TypeError):
                        continue
            if not date_column:
                raise ValueError("No datetime column found in the dataset.")
            print(date_column)
            # Set the date column as index
            data[date_column] = pd.to_datetime(data[date_column])
            data.set_index(date_column, inplace=True)
            print(data.head(15))
            # Identify forecast columns (numeric columns)
            forecast_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if not forecast_columns:
                raise ValueError("No numeric columns found for forecasting in the dataset.")

            # Infer frequency of datetime index
            freq = pd.infer_freq(data.index)
            print(date_column, freq)
            # Determine m based on inferred frequency
            # Determine m based on inferred frequency
            if freq == '15T':  # Quarter-hourly data (every 15 minutes)
                m = 96  # Daily seasonality (96 intervals in a day)
            elif freq == '30T':  # Half-hourly data (every 30 minutes)
                m = 48  # Daily seasonality (48 intervals in a day)
            elif freq == 'H':  # Hourly data
                m = 24  # Daily seasonality (24 intervals in a day)
            elif freq == 'D':  # Daily data
                m = 7  # Weekly seasonality (7 days in a week)
            elif freq == 'W':  # Weekly data
                m = 52  # Yearly seasonality (52 weeks in a year)
            elif freq == 'M':  # Monthly data
                m = 12  # Yearly seasonality (12 months in a year)
            elif freq == 'Q':  # Quarterly data
                m = 4  # Yearly seasonality (4 quarters in a year)
            elif freq == 'A' or (freq and freq.startswith('A-')):  # Annual data (any month-end)
                m = 1  # No further seasonality within a year
            else:
                raise ValueError(f"Unsupported frequency '{freq}'. Ensure data is in a common time interval.")
            results = {}
            try:
                data_actual = data[target_col].dropna()  # Remove NaNs if any

                # Split data into train and test sets
                train = data_actual.iloc[:-m]
                test = data_actual.iloc[-m:]

                # Auto ARIMA model selection
                model = pm.auto_arima(train,
                                      m=m,  # frequency of seasonality
                                      seasonal=True,  # Enable seasonal ARIMA
                                      d=None,  # determine differencing
                                      test='adf',  # adf test for differencing
                                      start_p=0, start_q=0,
                                      max_p=12, max_q=12,
                                      D=None,  # let model determine seasonal differencing
                                      trace=True,
                                      error_action='ignore',
                                      suppress_warnings=True,
                                      stepwise=True)
                # Forecast and calculate errors
                fc, confint = model.predict(n_periods=m, return_conf_int=True)
                # Save results to dictionary
                results = {
                    "actual": {
                        "date": list(test.index.astype(str)),
                        "values": [float(val) if isinstance(val, np.float_) else int(val) for val in
                                   test.values]
                    },
                    "forecast": {
                        "date": list(test.index.astype(str)),
                        "values": [float(val) if isinstance(val, np.float_) else int(val) for val in fc]
                    }
                }
                if not os.path.exists(os.path.join("models", 'arima', target_col)):
                    os.makedirs(os.path.join("models", 'arima', target_col), exist_ok=True)
                with open(os.path.join("models", 'arima', target_col, target_col + '_results.json'), 'w') as fp:
                    json.dump(results, fp)
                    plot_graph(results, os.path.join('models', 'arima', target_col))
                print(f"Results saved to {os.path.join('models', 'arima', target_col, target_col + '_results.json')}")
            except Exception as e:
                print(e)
                return False
            return True
    except Exception as e:
        print(e)
        return False


def plot_graph(data, file_path):
    try:
        col = file_path.split('\\')[-1]
        actual_dates = [datetime.strptime(date, "%Y-%m-%d") for date in data["actual"]["date"]]
        forecast_dates = [datetime.strptime(date, "%Y-%m-%d") for date in data["forecast"]["date"]]

        # Extract values
        actual_values = data["actual"]["values"]
        forecast_values = data["forecast"]["values"]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(actual_dates, actual_values, label='Actual', color='blue', marker='o')
        plt.plot(forecast_dates, forecast_values, label='Forecast', color='orange', linestyle='--', marker='x')

        # Formatting
        plt.title(f'{col} Actual vs Forecast Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the plot as a PNG image
        plt.savefig(os.path.join(file_path, "actual_vs_forecast.png"), format="png", dpi=300)
    except Exception as e:
        print(e)


def kmeans_train(data):
    try:
        # Identify categorical and numerical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Handle missing values (if any)
        imputer = SimpleImputer(strategy='mean')
        data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
        joblib.dump(imputer, 'imputer.pkl')

        # Build a transformer for preprocessing: scaling numerical columns and encoding categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),  # Standard scaling for numerical columns
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                # One-Hot encoding for categorical columns
            ])

        # Apply preprocessing and fit KMeans
        X = preprocessor.fit_transform(data)

        # Find the optimal k using the elbow method with KMeans
        inertia = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Determine the optimal k
        optimal_k = find_elbow_point(inertia)
        print('Optimal number of clusters (k) based on the Elbow Method:', optimal_k)

        # Initialize KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)

        # Fit KMeans to the preprocessed data
        kmeans.fit(X)

        # Save the trained model and preprocessor
        joblib.dump(kmeans, 'kmeans_model.pkl')  # Save KMeans model
        joblib.dump(preprocessor, 'preprocessor.pkl')  # Save Preprocessing pipeline

        # Add cluster labels to the original data
        data['Cluster'] = kmeans.labels_
        return True, data
    except Exception as e:
        print(e)
        return False, data


def load_pipeline(save_path="model_pipeline.pkl"):
    # Load the saved pipeline
    pipeline = joblib.load(save_path)
    print(f"Pipeline loaded from: {save_path}")
    return pipeline

def random_forest(data, target_column):
    try:
        if not os.path.exists(os.path.join("models", "rf", target_column, 'deployment.json')):
            os.makedirs(os.path.join("models", "rf", target_column),exist_ok=True)
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Detect categorical and numerical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

            # Preprocessing pipelines for numerical and categorical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])

            # Choose Random Forest type based on target type
            if y.nunique() <= 5:  # Classification for few unique target values
                model_type='Classification'
                model = RandomForestClassifier(random_state=42)
            else:  # Regression for continuous target values
                model_type='Regression'
                model = RandomForestRegressor(random_state=42)

            # Create pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            cv = min(5, len(X_test))

            # Evaluate the model using cross-validation
            scores = cross_val_score(pipeline, X_test, y_test, cv=cv)
            print(f"Model Performance (CV): {scores.mean():.4f} ± {scores.std():.4f}")

            # Save the pipeline
            joblib.dump(pipeline, os.path.join("models", "rf", target_column, "pipeline.pkl"))
            print(f'Pipeline saved to: {os.path.join("models", "rf", target_column, "pipeline.pkl")}')

            with open(os.path.join("models", "rf", target_column, "deployment.json"), "w") as fp:
                json.dump({"columns": list(X_train.columns), "model_type": model_type, "Target_column": target_column}, fp, indent=4)
            return True, list(X_train.columns)
        else:
            with open(os.path.join(os.getcwd(), "models", "rf", target_column, 'deployment.json'),"r") as fp:
                data = json.load(fp)
            return True, data['columns']
    except Exception as e:
        print(e)
        return False, []


@csrf_exempt
def model_predict(request):
    try:
        if request.POST.get('form_name') == 'rf':
            res = {}
            for col in request.POST:
                res.update({col: request.POST[col]})
            del res['form_name']
            df = pd.DataFrame([res])
            loaded_pipeline = load_pipeline(os.path.join("models", "rf", request.session['col_predict'], "pipeline.pkl"))
            predictions = loaded_pipeline.predict(df)
            print(predictions)
            request.session['rf_result'] = predictions[0]
            return redirect('models')
    except Exception as e:
        print(e)


def load_models(model_type, df, target_col):
    try:
        if model_type == 'rf':
            model = load_model(os.path.join(os.getcwd(), 'models', "rf", target_col, "model.h5"))
            with open(os.path.join(os.getcwd(), 'models', "rf", target_col, "deployment.json"), 'r') as fp:
                deployment_data = json.load(fp)
            for column in deployment_data["columns"]:
                if isinstance(deployment_data["columns"][column], list):
                    encoder_path = os.path.join(os.getcwd(), 'models', "rf", target_col,
                                                f'{column.replace(" ", "_")}_encoder.pkl')
                    df[column.replace("_", " ")] = joblib.load(encoder_path).fit_transform(df[column.replace("_", " ")])
                else:
                    df[column] = df[column].astype(float)
            res = model.predict(df.iloc[0, :].to_numpy().reshape(1, -1))
            model_type = deployment_data["model_type"]
            if model_type == 'classification':
                result = np.argmax(res, axis=-1)
                res = joblib.load(
                    os.path.join(os.getcwd(), 'models', "rf", target_col,
                                 f'{deployment_data["Target_column"].replace(" ", "_")}_encoder.pkl')).inverse_transform(
                    result)
            return res[0]

    except Exception as e:
        print(e)


def checkSentiment(df, categorical):
    sentiment = 'N'
    for i in categorical:
        # print([j for j in df[i]])
        data = ' '.join([str(j) for j in df[list(i.keys())[0]]]).upper()
        if ('GOOD' in data) | ('BAD' in data) | ('Better' in data):
            sentiment = "Y"
    return sentiment


def plot_numeric(numeric_vars, dataframe, line_plot_df=None):
    df = dataframe
    for i in list(numeric_vars.keys()):
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.patch.set_facecolor('White')
        plt.axhline(y=df[i].min(), color='b', linestyle='--', label='min')
        plt.axhline(y=df[i].max(), color='r', linestyle='--', label='max')
        plt.axhline(y=df[i].median(), color='g', linestyle='--', label='median')

        sns.barplot(x=np.arange(len(df[i])), y=i,
                    data=df)
        ax.set_ylabel(i, fontsize=15)
        ax.set_title(i, fontsize=20)
        ax.set_xticks(np.arange(len(df)))
        # ax.set_xticklabels(df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'), rotation=45, ha='right')
        plt.legend()
        plt.savefig(
            os.path.join(os.getcwd(), 'static/plots/bar/Bar_' + i.replace(' ', '').replace('/', '') + '.png'))


def missingvalueplots(names, missingvalue):
    i = 0
    for col in missingvalue.keys():
        if len(missingvalue[col].values()) > 0:
            fig, ax = plt.subplots(figsize=(20, 10))
            fig.patch.set_facecolor('White')
            df = pd.DataFrame({"Par": list(missingvalue[col].keys()), "Val": list(missingvalue[col].values())})
            # plt.scatter(list(missingvalue[col].keys()),list(missingvalue[col].values()),s=1000)
            sns.barplot(x='Par', y="Val", data=df)
            ax.set_ylabel('Count', fontsize=15)
            ax.set_xlabel(col, fontsize=15)
            ax.set_title(col, fontsize=20)
            plt.savefig(col.replace(' ', '').replace('/', '') + '.png')
            plt.savefig(os.path.join(os.getcwd(),
                                     'static/plots/missingValues/Bar_' + col.replace(' ', '').replace('/',
                                                                                                      '') + '.png'))
            i += 1


def missingvalueplot2(df, data):
    dfdtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    for j in data:
        if j:
            if df[j].isnull().sum() > 0:
                if dfdtypes[j] == 'object':
                    uniqueval = df[j].unique().tolist()[:-1]
                    mapdata = {j: i + 1 for i, j in enumerate(uniqueval)}
                    df[j] = df[j].map(mapdata)
                fig, ax = plt.subplots(figsize=(20, 10))
                sns.barplot(x=np.arange(len(df[j])), y=j, data=df)
                nullvalues = df[df[j].isnull()].index.tolist()
                plt.scatter(nullvalues, [0] * len(nullvalues), s=1000)
                ax.set_xticks(np.arange(len(df[j])))
                ax.set_title(j)
                plt.savefig(
                    os.path.join(os.getcwd(), 'static/plots/missingValues/Bar2_' + j.replace(' ', '').replace('/',
                                                                                                              '') + '.png'))


def plot_categorical(categorial_vars, dataframe):
    df = dataframe
    for i in categorial_vars:
        name = [k[0] for k in df[list(i)].value_counts().index.tolist()]
        count = df[list(i)].value_counts().values.tolist()
        fig, ax = plt.subplots(figsize=(10, 5))
        palette_color = sns.color_palette('bright')
        plt.pie(count, labels=name, colors=palette_color, autopct='%.0f%%')
        fig.patch.set_facecolor('White')
        ax.set_title(list(i)[0], fontsize=10)
        plt.savefig(os.path.join(os.getcwd(),
                                 'static/plots/pie/Pie_' + list(i)[0].replace(' ', '').replace('/', '') + '.png'))


def plot_wordCloud(text_data, dataframe):
    df = dataframe
    for i in text_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        text = " ".join(cat for cat in df[i])
        wordcloud = WordCloud(collocations=False, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        fig.patch.set_facecolor('White')
        ax.set_title(i, fontsize=10)
        plt.grid(False)
        plt.axis('off')
        plt.savefig(os.path.join(os.getcwd(),
                                 'static/plots/wordCloud/wordCloud_' + i.replace(' ', '').replace('/',
                                                                                                  '') + '.png'))


def adf_test(df, kpi):
    df_t = df.set_index(kpi)

    for col in df_t.columns:
        # Check if the column name is not in the specified list and is numeric
        if col.upper() not in ['DATE', 'TIME', 'DATE_TIME'] and pd.api.types.is_numeric_dtype(df_t[col]):
            dftest = adfuller(df_t[col], autolag='AIC')
            statistic_value = dftest[0]
            p_value = dftest[1]
            if (p_value > 0.5) and all([statistic_value > j for j in dftest[4].values()]):
                return "Y"

    return "N"
