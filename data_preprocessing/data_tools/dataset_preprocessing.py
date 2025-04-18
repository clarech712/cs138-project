import pandas as pd

class DataFrameSummarizer:
    """
    A class to provide a detailed summary and overview of a Pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataFrameSummarizer with a Pandas DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be summarized.
        """
        self.df = df
        self.shape = df.shape
        self.size = df.size
        self.columns = df.columns
        self.dtypes = df.dtypes
        self.na_counts = df.isna().sum()
        self.memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # in MB

    def display_overview(self):
        """
        Prints a high-level overview of the DataFrame.
        """
        print("=" * 40)
        print("DataFrame Overview")
        print("=" * 40)
        print(f"Shape: {self.shape}")
        print(f"Size: {self.size}")
        print(f"Number of Columns: {len(self.columns)}")
        print(f"Memory Usage: {self.memory_usage:.2f} MB")
        print("\nColumns:")
        for col in self.columns:
            print(f"- {col}")
        print("=" * 40)

    def display_column_details(self):
        """
        Prints detailed information about each column in the DataFrame.
        """
        print("\n" + "=" * 40)
        print("Column Details")
        print("=" * 40)
        for col in self.columns:
            print(f"\nColumn: {col}")
            print(f"  - Data Type: {self.dtypes[col]}")
            print(f"  - Number of Missing Values: {self.na_counts[col]}")
            print(f"  - Percentage of Missing Values: {(self.na_counts[col] / len(self.df) * 100):.2f}%")

            if self.dtypes[col] == 'object':  # Categorical or String
                try:
                    unique_values = self.df[col].nunique()
                    print(f"  - Number of Unique Values: {unique_values}")
                    if unique_values < 20:  # Display unique values if few enough
                        print(f"  - Unique Values: {self.df[col].unique()}")
                except:
                    print("  - Could not determine unique values.")
            else:  # Numerical
                try:
                    print(f"  - Minimum Value: {self.df[col].min()}")
                    print(f"  - Maximum Value: {self.df[col].max()}")
                    print(f"  - Mean Value: {self.df[col].mean()}")
                    print(f"  - Standard Deviation: {self.df[col].std()}")
                except:
                    print("  - Could not determine min/max/mean/std.")
        print("=" * 40)

    def display_summary(self):
        """
        Combines the overview and column details for a complete summary.
        """
        self.display_overview()
        self.display_column_details()

    def get_summary(self):
        """
        Returns a string containing the summary of the DataFrame.
        """
        import io
        buffer = io.StringIO()

        print("=" * 40, file=buffer)
        print("DataFrame Overview", file=buffer)
        print("=" * 40, file=buffer)
        print(f"Shape: {self.shape}", file=buffer)
        print(f"Size: {self.size}", file=buffer)
        print(f"Number of Columns: {len(self.columns)}", file=buffer)
        print(f"Memory Usage: {self.memory_usage:.2f} MB", file=buffer)
        print("\nColumns:", file=buffer)
        for col in self.columns:
            print(f"- {col}", file=buffer)
        print("=" * 40, file=buffer)

        print("\n" + "=" * 40, file=buffer)
        print("Column Details", file=buffer)
        print("=" * 40, file=buffer)
        for col in self.columns:
            print(f"\nColumn: {col}", file=buffer)
            print(f"  - Data Type: {self.dtypes[col]}", file=buffer)
            print(f"  - Number of Missing Values: {self.na_counts[col]}", file=buffer)
            print(f"  - Percentage of Missing Values: {(self.na_counts[col] / len(self.df) * 100):.2f}%", file=buffer)

            if self.dtypes[col] == 'object':  # Categorical or String
                try:
                    unique_values = self.df[col].nunique()
                    print(f"  - Number of Unique Values: {unique_values}", file=buffer)
                    if unique_values < 20:  # Display unique values if few enough
                        print(f"  - Unique Values: {self.df[col].unique()}", file=buffer)
                except:
                    print("  - Could not determine unique values.", file=buffer)
            else:  # Numerical
                try:
                    print(f"  - Minimum Value: {self.df[col].min()}", file=buffer)
                    print(f"  - Maximum Value: {self.df[col].max()}", file=buffer)
                    print(f"  - Mean Value: {self.df[col].mean()}", file=buffer)
                    print(f"  - Standard Deviation: {self.df[col].std()}", file=buffer)
                except:
                    print("  - Could not determine min/max/mean/std.", file=buffer)
        print("=" * 40, file=buffer)

        return buffer.getvalue()
