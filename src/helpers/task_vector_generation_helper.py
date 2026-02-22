import pandas as pd
from collections import defaultdict
import re
import numpy as np
from torch_frame import stype
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from prince import MCA
from src.helpers.database_helper import remove_unnamed_columns, drop_nan_columns
import warnings
from src.utils.progress_parallel import ProgressParallel
from joblib import delayed

def get_task_columns_dict(db, stype_dict):
    # Initialize dictionaries
    not_needed_columns = defaultdict(list)
    task_columns_dict = {}
    
    # Get primary and foreign key columns
    for table_name, table in db.table_dict.items():
        not_needed_columns[table_name].append(table.pkey_col)
        not_needed_columns[table_name].extend(table.fkey_col_to_pkey_table.keys())
    
    # Get special type columns that should be excluded
    special_types = ['text_embedded', 'timestamp', 'multicategorical']
    for table_name, column_dict in stype_dict.items():
        for col_name, col_type in column_dict.items():
            if any(col_type == stype(t) for t in special_types):
                not_needed_columns[table_name].append(col_name)
                
    # Get remaining columns for each table
    for table_name, column_dict in stype_dict.items():
        task_columns_dict[table_name] = [
            col for col in column_dict.keys() 
            if col not in not_needed_columns[table_name]
        ]
        
    return task_columns_dict

def generate_task_vector_clauses(path, pkey_col, task_columns_dict, stype_dict, postfix=0, cols=None, exclude_mode_cols=None):
    """
    Generate sub-query clauses for task vector generation.
    
    Args:
        path: List of table names in the path
        pkey_col: Primary key column name of the first table in the path
        task_columns_dict: Dictionary of task columns for each table
        stype_dict: Dictionary of stype for each table
        postfix: Postfix for the path name
        
    Returns:
        sub_query_select_clauses: Sub-query select clauses
        final_select_clauses: Final select clauses
        when_clause: When clause
    """
    path_name = "_".join(path)
    path_name += f"_{postfix}"
    query_cols = []
    for col in task_columns_dict[path[-1]]:
        if cols is not None and col not in cols:
            continue
        query_cols.append(f"{path[-1]}__{path.count(path[-1]) - 1}.{col}")
    assert query_cols, "No columns to be selected."

    numerical_cols = []
    categorical_cols = []
    col_names = []
    for col in query_cols:
        table_idx, col = col.split('.')
        table, idx = table_idx.rsplit('__', 1)
        if stype_dict[table][col] == stype('numerical'):
            numerical_cols.append(f"MEAN({table}__{idx}.{col}) as mean__{table}__{idx}__{col}")
            col_names.append(f"mean__{table}__{idx}__{col}")
            numerical_cols.append(f"MIN({table}__{idx}.{col}) as min__{table}__{idx}__{col}")
            col_names.append(f"min__{table}__{idx}__{col}")
            numerical_cols.append(f"MAX({table}__{idx}.{col}) as max__{table}__{idx}__{col}")
            col_names.append(f"max__{table}__{idx}__{col}")
            numerical_cols.append(f"SUM({table}__{idx}.{col}) as sum__{table}__{idx}__{col}")
            col_names.append(f"sum__{table}__{idx}__{col}")
            numerical_cols.append(f"COUNT({table}__{idx}.{col}) as count__{table}__{idx}__{col}")
            col_names.append(f"count__{table}__{idx}__{col}")
            numerical_cols.append(f"STDDEV({table}__{idx}.{col}) as stddev__{table}__{idx}__{col}")
            col_names.append(f"stddev__{table}__{idx}__{col}")
        elif stype_dict[table][col] == stype('categorical'):
            if exclude_mode_cols is not None and col in exclude_mode_cols:
                categorical_cols.append(f"COUNT({table}__{idx}.{col}) as count__{table}__{idx}__{col}")
                col_names.append(f"count__{table}__{idx}__{col}")
                categorical_cols.append(f"COUNT(DISTINCT {table}__{idx}.{col}) as count_distinct__{table}__{idx}__{col}")
                col_names.append(f"count_distinct__{table}__{idx}__{col}")
            else:
                categorical_cols.append(f"MODE({table}__{idx}.{col}) as mode__{table}__{idx}__{col}")
                col_names.append(f"mode__{table}__{idx}__{col}")
                categorical_cols.append(f"COUNT({table}__{idx}.{col}) as count__{table}__{idx}__{col}")
                col_names.append(f"count__{table}__{idx}__{col}")
                categorical_cols.append(f"COUNT(DISTINCT {table}__{idx}.{col}) as count_distinct__{table}__{idx}__{col}")
                col_names.append(f"count_distinct__{table}__{idx}__{col}")
    select_numerical_clauses = ",\n".join(numerical_cols)
    select_categorical_clauses = ",\n".join(categorical_cols)
    if select_numerical_clauses and select_categorical_clauses:
        sub_query_select_clauses = select_numerical_clauses + ",\n" + select_categorical_clauses
    elif select_numerical_clauses:
        sub_query_select_clauses = select_numerical_clauses
    elif select_categorical_clauses:
        sub_query_select_clauses = select_categorical_clauses
    final_select_clauses = [f"{path_name}.{col} as {path_name}__{col}" for col in col_names]
    final_select_clauses = ",\n".join(final_select_clauses)
    when_clause = f"{path_name}.{path[0]}__0__{pkey_col} IS NOT NULL"
    
    return sub_query_select_clauses, final_select_clauses, when_clause

def sanitize_column_name(name):
    """
    Converts a category name into a valid SQL column name.
    - Replaces non-alphanumeric characters with underscores.
    - Ensures the column name does not start with a number by adding a leading underscore.
    """
    name = re.sub(r'[^\w]', '_', name)  # Replace all non-word characters (A-Z, a-z, 0-9, _) with "_"
    name = re.sub(r'_+', '_', name)  # Collapse multiple underscores to a single "_"
    name = name.strip('_')  # Remove leading/trailing underscores
    
    # Ensure column name does not start with a number
    if re.match(r'^\d', name):
        name = '_' + name  # Prefix with underscore if it starts with a number

    return name

def transform_features(df, std_scalers=None, onehot_encoder=None, mca_encoder=None, is_train=True, mca_num_components=None):
    """
    Transform features using StandardScaler for numerical and OneHotEncoder for categorical columns.
    
    Args:
        df: Input dataframe
        std_scalers: Dict of StandardScaler objects for each numerical column, or None to create new
        onehot_encoder: OneHotEncoder object, or None to create new
        mca_encoder: MCA object, or None to create new
        is_train: If True, fit and transform. If False, only transform using provided scalers/encoder
    
    Returns:
        transformed_df: Transformed dataframe
        std_scalers: Dict of fitted StandardScaler objects
        onehot_encoder: Fitted OneHotEncoder object
    """
    categorical_cols = [col for col in df.columns if '__mode__' in col]
    print("Categorical columns:")
    print(categorical_cols)
    numerical_cols = [col for col in df.columns[2:-1] if col not in categorical_cols]
    print("Numerical columns:")
    print(numerical_cols)
    label_col = df.columns[-1]
    
    # Initialize or use provided scalers
    if std_scalers is None:
        std_scalers = {col: MinMaxScaler() for col in numerical_cols}
    if onehot_encoder is None and mca_num_components is None:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if mca_encoder is None and mca_num_components is not None:
        mca_encoder = MCA(n_components=mca_num_components, n_iter=50, copy=True, check_input=True, engine="sklearn", random_state=42)
        
    # Create copy of dataframe for scaling
    transformed_df = df[df.columns[:-1]].copy()
    
    # Scale numerical columns
    for col in numerical_cols:
        non_nan_mask = df[col].notna()
        non_nan_data = df[col][non_nan_mask].values.reshape(-1, 1)
        if is_train:
            scaled_data = std_scalers[col].fit_transform(non_nan_data)
        else:
            scaled_data = std_scalers[col].transform(non_nan_data)
        transformed_df[col] = transformed_df[col].astype(float)
        transformed_df.loc[non_nan_mask, col] = scaled_data.flatten().astype(float)
        transformed_df.loc[~non_nan_mask, col] = 0.0
    
    # One-hot encode categorical columns
    if categorical_cols:
        if mca_num_components is None:
            # Convert categorical data to string
            cat_data = transformed_df[categorical_cols].copy()
            # for col in categorical_cols:
            #     cat_data[col] = cat_data[col].astype(str)
            
            # Fit/transform or transform only
            if is_train:
                onehot_encoded = onehot_encoder.fit_transform(cat_data)
            else:
                onehot_encoded = onehot_encoder.transform(cat_data)
            
            # Get feature names
            feature_names = []
            for i, col in enumerate(categorical_cols):
                categories = onehot_encoder.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            # Convert to DataFrame and fill with zeros for unknown categories
            onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=transformed_df.index)
            
            # Drop original categorical columns and join encoded ones
            transformed_df = transformed_df.drop(columns=categorical_cols)
            transformed_df = pd.concat([transformed_df, onehot_df], axis=1)
        else:
            cat_data = transformed_df[categorical_cols].copy()
            if is_train:
                mca_encoder = mca_encoder.fit(cat_data)
            mca_encoded = mca_encoder.transform(cat_data)
            cols = list(mca_encoded.columns.values)
            mca_df = pd.DataFrame(mca_encoded, columns=cols, index=transformed_df.index)
            
            # Drop original categorical columns and join encoded ones
            transformed_df = transformed_df.drop(columns=categorical_cols)
            transformed_df = pd.concat([transformed_df, mca_df], axis=1)
    
    transformed_df[label_col] = df[label_col]
    
    return transformed_df, std_scalers, onehot_encoder, mca_encoder

def get_timestamps_for_split(dataset, stype_dict, split, timedelta, num_eval_timestamps):
    """Get timestamps for a given split (train/val/test).
    
    Args:
        dataset: Dataset object containing timestamp information
        split: String indicating split ('train', 'val', or 'test')
        timedelta: pandas.Timedelta object for time intervals
        num_eval_timestamps: Number of evaluation timestamps
        
    Returns:
        timestamps: pandas.DatetimeIndex of timestamps for the split
    """
    db = dataset.get_db(upto_test_timestamp=split != "test")
    for table_name, table in db.table_dict.items():
        df = table.df
        remove_unnamed_columns(df)                         # Remove unnamed columns
        drop_nan_columns(df)                               # Drop columns with all NaN values
        for col in df.columns:                                  # Check if all columns are in stype proposal
            assert col in stype_dict[table_name].keys(), f"Column {col} not in stype proposal."

    if split == "train":
        start = dataset.val_timestamp - timedelta
        end = db.min_timestamp
        freq = -timedelta

    elif split == "val":
        if dataset.val_timestamp + timedelta > db.max_timestamp:
            raise RuntimeError(
                "val timestamp + timedelta is larger than max timestamp! "
                "This would cause val labels to be generated with "
                "insufficient aggregation time."
            )

        start = dataset.val_timestamp
        end = min(
            dataset.val_timestamp
            + timedelta * (num_eval_timestamps - 1),
            dataset.test_timestamp - timedelta,
        )
        freq = timedelta

    elif split == "test":
        if dataset.test_timestamp + timedelta > db.max_timestamp:
            raise RuntimeError(
                "test timestamp + timedelta is larger than max timestamp! "
                "This would cause test labels to be generated with "
                "insufficient aggregation time."
            )

        start = dataset.test_timestamp
        end = min(
            dataset.test_timestamp
            + timedelta * (num_eval_timestamps - 1),
            db.max_timestamp - timedelta,
        )
        freq = timedelta

    timestamps = pd.date_range(start=start, end=end, freq=freq)
    return db, timestamps
