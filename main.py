from dataclasses import dataclass
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

@dataclass
class SummaryStats:
    """A data class to store summary statistics."""
    table: pd.DataFrame

@dataclass    
class DiffTable:
    """A data class to store difference tables."""
    table: pd.DataFrame
    
@dataclass
class RegressionResults:
    """A data class to store regression results."""
    table: pd.DataFrame


class DataAnalyzer:
    """
    A class to perform analysis on a given dataset. It can generate summary statistics,
    difference tables and regression results.

    Attributes
    ----------
    data : pd.DataFrame
        The dataframe to be analyzed.

    Methods
    -------
    create_summary_stats(variables: List[str], groups: List[str], level: str = None) -> SummaryStats:
        Generates a summary statistics table for the given variables and groups.
    create_diff_table(outcome: str, groups: List[str], level: str = None) -> DiffTable:
        Generates a difference table for the given outcome and groups.
    regression_results(outcome: str, covariates: List[str], level: str = None) -> RegressionResults:
        Performs a regression analysis and returns the results.
    """
    def __init__(self, data: pd.DataFrame):
        """Initializes the DataAnalyzer with a dataset."""
        self.data = data

    def create_summary_stats(self, variables: list, groups: list, level: str = None) -> SummaryStats:
        """
        Generates a summary statistics table for the given variables and groups.

        Parameters
        ----------
        variables : List[str]
            The list of variable names to include in the summary.
        groups : List[str]
            The list of group names to include in the summary.
        level : str, optional
            The level to groupby in the dataframe, by default None.

        Returns
        -------
        SummaryStats
            The resulting SummaryStats object containing the summary table.
        """
        if level:
            table = self.data.groupby(level)[variables].agg(['mean', 'std']).reset_index()
        else: 
            table = self.data[variables].agg(['mean', 'std'])

        table.columns = [f'{v}_{s}' for v, s in table.columns]
        
        return SummaryStats(table)

    def create_diff_table(self, outcome: str, groups: list, level: str = None) -> DiffTable:
        """
        Generates a difference table for the given outcome and groups.

        Parameters
        ----------
        outcome : str
            The outcome variable to consider.
        groups : List[str]
            The list of group names to include in the table.
        level : str, optional
            The level to groupby in the dataframe, by default None.

        Returns
        -------
        DiffTable
            The resulting DiffTable object containing the difference table.
        """
        if level:
            means = self.data.groupby([level, *groups])[outcome].mean().unstack(level=-1)
        else:
            means = self.data.groupby(groups)[outcome].mean()

        diff = means.subtract(means.values, axis=1)
        
        return DiffTable(diff)

    def regression_results(self, outcome: str, covariates: list, level: str = None) -> RegressionResults:
        """
        Performs a regression analysis and produces the results.

        Parameters
        ----------
        outcome : str
            The outcome variable to consider.
        covariates : List[str]
            The list of covariate names to include in the regression.
        level : str, optional
            The level to groupby in the dataframe, by default None.

        Returns
        -------
        RegressionResults
            The resulting RegressionResults object containing the regression table.
        """
        if level:
            X = self.data.groupby(level)[covariates].mean().reset_index()
            y = self.data.groupby(level)[outcome].mean().reset_index()[outcome] 
        else:
            X = self.data[covariates]
            y = self.data[outcome]

        X = sm.add_constant(X)
        if X.isnull().values.any():
            raise ValueError("Data contains NaN values. Please clean your data before running the regression.")
            
        model = sm.OLS(y, X)
        results = model.fit()
        
        metrics = {
            'r_squared': results.rsquared,
            'rmse': mean_squared_error(y, results.predict(X), squared=False)        
        }
        
        table = results.summary().tables[1].append(pd.DataFrame(metrics, index=['r_squared', 'rmse']))
        
        return RegressionResults(table)
