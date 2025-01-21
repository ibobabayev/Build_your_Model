import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR,SVC
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBRegressor,XGBClassifier
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


st.set_page_config(
    page_title="Build your model",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    option = st.selectbox(
        "",
        ("Homepage", "Preprocessing", "Modeling", "Evaluation and Visualization"), )

st.subheader("This project involves predicting outcomes using a simple model. It includes steps such as data preprocessing, model building, and concludes with evaluating the model's performance through metrics and visualizations.")

if option == 'Homepage':
    uploaded_file = st.file_uploader(
        "Upload your CSV file :file_folder:")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        st.caption("Let's recap first 5 observations of the dataset")
        st.write(data.head())
        st.caption("Data Types of Columns")
        st.write(data.dtypes)
        st.markdown('Once you are familiar with the data, you can start **preprocessing**')

elif option == 'Preprocessing':
    st.caption("Number of NA values in each column")
    st.write(st.session_state.data.isnull().sum())

    numerical_columns = st.session_state.data.select_dtypes(include=['number']).columns

    null_numerical_columns = []
    for col in numerical_columns:
        if st.session_state.data[col].isnull().sum() > 0:
            null_numerical_columns.append(col)

    fill_num = st.toggle("Would you like to fill NA values in numerical columns?")
    if fill_num:
        chosen_columns_num = st.multiselect(
            "Choose the columns you want to fill",
            [col for col in numerical_columns if st.session_state.data[col].isnull().sum() > 0],
            placeholder="Choose columns", default=None,
            key=f'{[col for col in numerical_columns if st.session_state.data[col].isnull().sum() > 0]}'
        )
        for col in chosen_columns_num:
            method_num = st.radio(
                f"Choose method for column '{col}'", ["Mean", "Median", "Constant value", "Delete the observation"],
                index=None, key=f"method_{col}"
            )
            if method_num == 'Mean':
                st.session_state.data[col] = round(st.session_state.data[col].fillna(st.session_state.data[col].mean()))

            elif method_num == 'Median':
                st.session_state.data[col] = round(
                    st.session_state.data[col].fillna(st.session_state.data[col].median()))

            elif method_num == 'Constant value':
                number = st.text_input(f"Insert a number to fill NA values in '{col}'", value="")
                if number != "":
                    try:
                        number = float(number)
                        st.session_state.data[col] = st.session_state.data[col].fillna(number)
                        st.success(f'All NA values in "{col}" changed to {number}')
                    except ValueError:
                        st.error("Please enter a valid number!")

            elif method_num == 'Delete the observation':
                st.session_state.data = st.session_state.data.dropna(subset=[col]).reset_index(drop=True)

    categorical_columns = st.session_state.data.select_dtypes(include=['object']).columns
    null_categorical_columns = []
    for col in categorical_columns:
        if st.session_state.data[col].isnull().sum() > 0:
            null_categorical_columns.append(col)
    fill_cat = st.toggle("Would you like to fill NA values in categorical columns?")

    if fill_cat:
        chosen_columns_cat = st.multiselect(
            "Choose the columns you want to fill",
            [col for col in categorical_columns if st.session_state.data[col].isnull().sum() > 0],
            placeholder="Choose columns", default=None,
            key=f"method_{col}"
        )
        for col in chosen_columns_cat:
            method_cat = st.radio(
                f"Choose method for column '{col}'", ["Mode", "Other", "Specific word", "Delete the observation"],
                key=f"method_{col}", index=None
            )
            if method_cat == 'Mode':
                st.session_state.data[col] = st.session_state.data[col].fillna(st.session_state.data[col].mode()[0])

            elif method_cat == 'Other':
                st.session_state.data[col] = st.session_state.data[col].fillna('Other')

            elif method_cat == 'Specific word':
                word = st.text_input(f"Type a word to fill NA values in '{col}'", value="")
                if word != "":
                    try:
                        st.session_state.data[col] = st.session_state.data[col].fillna(word)
                        st.success(f'All NA values in "{col}" changed to {word}')
                    except ValueError:
                        st.error("Please enter a valid word!")

            elif method_cat == "Delete the observation":
                st.session_state.data = st.session_state.data.dropna(subset=[col]).reset_index(drop=True)

    column_to_analyze = st.selectbox("Select a column to see a basic statistics", numerical_columns)
    if column_to_analyze:
        st.subheader(f"Analyzing: {column_to_analyze}")
        st.write(st.session_state.data[column_to_analyze].describe())

    st.caption("Let's take a look at the distributions of the columns")
    num_col_name = st.selectbox("Choose a column", numerical_columns)
    fig = plt.figure(figsize=(16, 6))
    sns.histplot(st.session_state.data[num_col_name], kde=True)
    plt.title(f'Distribution of {num_col_name}')
    st.pyplot(fig)

    st.caption("Let's take a look at the outliers")
    num_col_name2 = st.selectbox("Choose a column", numerical_columns, key='Unique')
    fig = plt.figure(figsize=(16, 6))
    sns.boxplot(x=st.session_state.data[num_col_name2])
    plt.title(f'Boxplot of {num_col_name2} to show outliers')
    st.pyplot(fig)

    with st.expander("See correlation matrix"):
        correlation_matrix = st.session_state.data.corr(numeric_only=True)
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        st.pyplot(fig)

    st.caption("Do your model need an encoding?")
    le_button = st.button("Do label encoding", type="primary")
    st.warning('Do not forget to fill NULL values in categorical columns')
    if le_button:
        le = LabelEncoder()
        for cat_col in categorical_columns:
            encoded = le.fit_transform(st.session_state.data[cat_col])
            st.session_state.data[cat_col + '_encoded'] = encoded

        st.write('You can see decoded values of encoding')
        st.write(st.session_state.data)

    st.markdown('After preprocessing the data, you can start **building the model**')

elif option == 'Modeling':
    st.caption('First we need to remove irrelevant columns')
    options = st.session_state.data.columns
    deleted_col = st.pills(
        "Choose the columns you want to delete", options, default=None, key="unique",
        selection_mode="multi")
    if deleted_col:
        st.session_state.data.drop(deleted_col, axis=1, inplace=True)
    st.write(st.session_state.data)

    X = st.multiselect("Select the feature columns",
                       [col for col in st.session_state.data.columns],
                       default=None, key="features")
    y = st.selectbox("Select the target column", [col for col in st.session_state.data.columns if col not in X],
                     index=None)
    st.write(f'Features: {X}')
    st.write(f'Target: {y}')
    X = st.session_state.data[X]
    y = st.session_state.data[y]
    X = np.array(X)
    y = np.array(y)
    st.caption('Now we can split our data into training and testing sets')
    test_size = st.slider("Select the proportion for the test set.", 0, 100, 20, 5) / 100
    random_state = st.number_input("Set the random state in the train-test split for reproducibility", min_value=0,
                                   max_value=4294967295, value=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        shuffle=True)
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.y_test = y_test
    st.session_state.y_train = y_train

    scaling = st.toggle("Would you like to scale the columns in the data?")
    if scaling:
        method_scale = st.radio(
            "Choose scaling method", ["Min-Max Scaling", "Standard Scaler", "Robust Scaler"],
            index=None, key=f"scaling"
        )
        if method_scale == 'Min-Max Scaling':
            minmax_scaler = MinMaxScaler()
            X_train_scaled = minmax_scaler.fit_transform(X_train)
            X_test_scaled = minmax_scaler.transform(X_test)
            st.session_state.X_train_scaled = X_train_scaled
            st.session_state.X_test_scaled = X_test_scaled

        elif method_scale == 'Standard Scaler':
            standard_scaler = StandardScaler()
            X_train_scaled = standard_scaler.fit_transform(X_train)
            X_test_scaled = standard_scaler.transform(X_test)
            st.session_state.X_train_scaled = X_train_scaled
            st.session_state.X_test_scaled = X_test_scaled

        elif method_scale == 'Robust Scaler':
            robust_scaler = RobustScaler()
            X_train_scaled = robust_scaler.fit_transform(X_train)
            X_test_scaled = robust_scaler.transform(X_test)
            st.session_state.X_train_scaled = X_train_scaled
            st.session_state.X_test_scaled = X_test_scaled
    st.caption('The columns have been scaled')

    model_type = st.radio("Choose the type of model", ["Regression", "Classification"], index=None)
    st.session_state.model_type = model_type

    if model_type == 'Regression':
        models_reg = ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor',
                      'Support Vector Regression(SVR)', 'KNeighbors Regressor', 'LGBMRegressor',
                      'XGBRegressor', 'CatBoost Regressor']
        model = st.multiselect("Select up to **three** models that you want to use for prediction",
                               models_reg, default=None, max_selections=3, key="model_reg")
        st.session_state.model = model
        model_dict = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Regression(SVR)": SVR(),
            "KNeighbors Regressor": KNeighborsRegressor(),
            "LGBMRegressor": LGBMRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoost Regressor": CatBoostRegressor()
        }
        st.session_state.model_dict = model_dict
        ready = st.button("Make Prediction")
        if ready:
            st.caption("The model has been predicted. Let's check the results on the **Evaluation and Metrics page**")


    elif model_type == 'Classification':
        models_cl = ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Naive Bayes',
                     'Support Vector Classifier(SVC)', 'KNeighbors Classifier', 'LGBMClassifier',
                     'XGBClassifier', 'CatBoost Classifier']

        model_dict = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree Classifier': DecisionTreeClassifier(),
            'Random Forest Classifier': RandomForestClassifier(),
            'Naive Bayes': GaussianNB(),
            'Support Vector Classifier(SVC)': SVC(),
            'KNeighbors Classifier': KNeighborsClassifier(),
            'LGBMClassifier': LGBMClassifier(),
            'XGBClassifier': XGBClassifier(),
            'CatBoost Classifier': CatBoostClassifier()
        }
        model = st.multiselect("Select up to **three** models that you want to use for prediction",
                               models_cl, default=None, max_selections=3, key="model_cl")
        st.session_state.model = model
        st.session_state.model_dict = model_dict
        ready = st.button("Make Prediction")
        if ready:
            st.caption("The model has been predicted. Let's check the results on the **Evaluation and Visualization** page")


elif option == 'Evaluation and Visualization':
        st.button('REFRESH')
        if st.session_state.model_type == 'Regression':
            chosen_models = []
            r2_results = []
            adj_r2_results = []
            rmse_results = []
            for m in st.session_state.model:
                model_name = st.session_state.model_dict[m]
                model = model_name.fit(st.session_state.X_train_scaled, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test_scaled)
                R2 = metrics.r2_score(st.session_state.y_test, y_pred)
                n = st.session_state.X.shape[0]
                p = st.session_state.X.shape[1]
                adjusted_R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
                rmse = metrics.root_mean_squared_error(st.session_state.y_test, y_pred)
                chosen_models.append(m)
                r2_results.append(R2)
                adj_r2_results.append(adjusted_R2)
                rmse_results.append(rmse)
            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]
            for i, col in enumerate(columns):
                with col:
                    st.header(f'**{chosen_models[i]}**')
                    st.write(f'R2: **{r2_results[i]}**')
                    st.write(f'Adjusted R2: **{adj_r2_results[i]}**')
                    st.write(f'Root Mean Squared Error: **{rmse_results[i]}**')

            fig = plt.figure(figsize=(10, 6))
            plt.bar(st.session_state.model, r2_results, color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'])
            for i, v in enumerate(r2_results):
                plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
            plt.xlabel("Models")
            plt.ylabel("R-Square")
            plt.title("Comparison of Model R-Squares")
            plt.ylim(0, 1)
            st.pyplot(fig)

            fig = plt.figure(figsize=(10, 6))
            plt.bar(st.session_state.model, rmse_results, color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'])
            for i, v in enumerate(rmse_results):
                plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
            plt.xlabel("Models")
            plt.ylabel("Root Mean Squared Error")
            plt.title("Comparison of Model's Root Mean Squared Errors")
            plt.ylim(0, 1)
            st.pyplot(fig)

        elif st.session_state.model_type == 'Classification':
            chosen_models = []
            cr_results = []
            as_results = []
            cm_list = []
            for m in st.session_state.model:
                model_name = st.session_state.model_dict[m]
                model = model_name.fit(st.session_state.X_train_scaled, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test_scaled)
                cr = metrics.classification_report(st.session_state.y_test, y_pred,output_dict=True)
                cr = pd.DataFrame(cr).transpose()
                acc_score = metrics.accuracy_score(st.session_state.y_test, y_pred)
                cm = metrics.confusion_matrix(st.session_state.y_test, y_pred)
                disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
                cm_list.append(disp)
                chosen_models.append(m)
                cr_results.append(cr)
                as_results.append(acc_score)

            for i in range(len(chosen_models)):
                st.header(f'**{chosen_models[i]}**')
                st.write(f'**Classification Report**')
                st.dataframe(cr_results[i])
                st.write(f'Accuracy Score: **{as_results[i]}**')

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_xlabel('Predicted Labels', fontsize=8)
                ax.set_ylabel('True Labels', fontsize=8)
                ax.set_title('Confusion Matrix', fontsize=10)
                for tick in ax.get_xticklabels():
                    tick.set_fontsize(8)
                for tick in ax.get_yticklabels():
                    tick.set_fontsize(8)
                cm_list[i].plot(ax=ax)
                st.pyplot(fig)

            fig = plt.figure(figsize=(10, 6))
            plt.bar(st.session_state.model, as_results, color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon'])
            for i, v in enumerate(as_results):
                plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
            plt.xlabel("Models")
            plt.ylabel("Accuracy")
            plt.title("Comparison of Model`s Accuracies")
            plt.ylim(0, 1)
            st.pyplot(fig)

        if st.button('Your model is ready'):
            st.balloons()