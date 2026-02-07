import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
class DataPreprocessor:

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Ожидается pandas.DataFrame, получен {type(df)}")
        self.df = df.copy()
        self.dropped = None
        self.original_df = df.copy()  # Сохраняем оригинальные данные
        self.categorical_cols = None  # Для хранения списка категориальных столбцов



    def remove_missing(self, threshold: float = 0.5,fill_type="mean"):

        if fill_type not in ['mean', 'median','mode']:
            raise ValueError("fill_type должен быть 'mean', 'median','mode'")

        df = self.df.copy()
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        total_rows = len(df)

        fill_percentage = (df[numeric_columns].count() / total_rows ).round(2)
        columns_to_drop = fill_percentage[fill_percentage < threshold].index.tolist()

        columns_to_fill = fill_percentage[fill_percentage >= threshold].index.tolist()

        self.dropped = df[columns_to_drop].copy()
        df_cleaned = df.drop(columns=columns_to_drop).copy()



        if fill_type == "mean":
            df_cleaned[columns_to_fill] =df_cleaned[columns_to_fill].fillna(df_cleaned[columns_to_fill].mean().round(2))
        elif fill_type == "median":
            df_cleaned[columns_to_fill] =df_cleaned[columns_to_fill].fillna(df_cleaned[columns_to_fill].median().round(2))
        elif fill_type == "mode":
            for col in columns_to_fill:
                mode_values = df_cleaned[col].mode()
                if not mode_values.empty:
                    mode_val = mode_values.iloc[0]
                    df_cleaned.fillna({col:mode_val}, inplace=True)
                else:
                    df_cleaned[col].fillna(0, inplace=True)



        self.df = df_cleaned.copy()

    def encode_categorical(self,categorical: list[str]):
        df = self.df.copy()

        missing_cols = [col for col in categorical if col not in df.columns]
        if missing_cols:
            print(f"Предупреждение: Столбцы {missing_cols} не найдены в DataFrame")
            categorical = [col for col in categorical if col in df.columns]
        if not categorical:
            print("Нет столбцов для кодирования")
            return
        encoded_df = pd.get_dummies(df, columns=categorical)


        self.df = encoded_df.copy()
    def normalize_numeric(self,method: str = 'minmax'):

        if method not in ['minmax', 'std']:
            raise ValueError("method должен быть 'minmax' или 'std'")

        df  =self.df.copy()
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        if method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        self.df = df.copy()

    def fit_transform(self,
                      threshold: float = 0.5,
                      fill_type: str = 'mean',
                      categorical_cols: list[str] = None,
                      normalize_method: str = 'minmax',
                      reset: bool = True) -> pd.DataFrame:

        if reset:
            self.df = self.original_df.copy()


        self.remove_missing(threshold=threshold, fill_type=fill_type)
        print(f"   После обработки пропусков: {self.df.shape}\n")


        self.encode_categorical(categorical_cols)
        print(f"   После кодирования: {self.df.shape}\n")


        self.normalize_numeric(method=normalize_method)
        print(f"   После нормализации: {self.df.shape}\n")


        return self.df.copy()

    def get_original_data(self) -> pd.DataFrame:
        return self.original_df.copy()

    def get_processed_data(self) -> pd.DataFrame:
        return self.df.copy()

