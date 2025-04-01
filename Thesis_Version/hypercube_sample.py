import numpy as np
import pandas as pd
from scipy.stats import qmc

def create_lhs_table_with_condition(
    always_dict,
    conditional_dict,
    fraction_true,
    n_samples,
    boolean_name="my_boolean",
    output_csv="lhs_samples.csv",
    random_seed=None
):
    """
    Create an LHS table with:
      1) A boolean parameter that is True in 'fraction_true' of samples.
      2) Conditional parameters that only apply when the boolean is True.
         Otherwise, they take a default value.

    Parameters
    ----------
    always_dict : dict
        Dictionary of always-used parameters and their [min, max], e.g.:
          {
            'length': [0.0, 10.0],
            'width':  [5.0, 15.0],
            ...
          }
    conditional_dict : dict
        Dictionary of parameters that apply only if boolean_name is True, e.g.:
          {
            'cond_param1': [1.0, 2.0],
            'cond_param2': [10.0, 20.0],
            ...
          }
    fraction_true : float
        Fraction of runs (between 0 and 1) for which boolean_name is True.
    n_samples : int
        Total number of samples to generate.
    default_values_conditional : dict
        Default values for the conditional parameters (same keys as conditional_dict)
        to be used when boolean_name is False, e.g.:
          {
            'cond_param1': 1.5,    # default
            'cond_param2': 15.0,   # default
          }
    boolean_name : str, optional
        Name for the boolean column in the output (default "my_boolean").
    output_csv : str, optional
        Path for saving the CSV file (default "lhs_samples.csv").
    random_seed : int, optional
        If provided, fixes the random seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the combined samples, including the boolean column
        and the conditional parameters (which are set to default when the boolean
        is False).
    """

    # -----------------
    # 1) Prep Samplers
    # -----------------
    if random_seed is not None:
        np.random.seed(random_seed)

    # Extract parameter names & bounds for the always-used parameters
    always_params = list(always_dict.keys())
    lower_always = np.array([always_dict[p][0] for p in always_params])
    upper_always = np.array([always_dict[p][1] for p in always_params])

    # Extract parameter names & bounds for the conditional parameters
    cond_params = list(conditional_dict.keys())
    lower_cond = np.array([conditional_dict[p][0] for p in cond_params])
    upper_cond = np.array([conditional_dict[p][1] for p in cond_params])

    # Create LHS samplers for each set
    sampler_always = qmc.LatinHypercube(d=len(always_params), seed=random_seed)
    sampler_cond = qmc.LatinHypercube(d=len(cond_params), seed=None)  # separate sampler

    # -----------------
    # 2) Generate LHS samples
    # -----------------
    raw_always = sampler_always.random(n_samples)  # shape (n_samples, len(always_params))
    raw_cond = sampler_cond.random(n_samples)      # shape (n_samples, len(cond_params))

    # Scale them to the actual ranges
    scaled_always = qmc.scale(raw_always, lower_always, upper_always)
    scaled_cond = qmc.scale(raw_cond, lower_cond, upper_cond)

    # -----------------
    # 3) Generate Boolean column
    # -----------------
    # fraction_true of them should be True. We can do this by random uniform draws:
    booleans = np.random.rand(n_samples) < fraction_true
    # booleans is an array of shape (n_samples,), with True in ~ fraction_true portion.

    # -----------------
    # 4) Build the DataFrame
    # -----------------
    #   Always-used parameters -> columns
    df_always = pd.DataFrame(scaled_always, columns=always_params)

    #   Conditional parameters -> columns
    df_cond = pd.DataFrame(scaled_cond, columns=cond_params)

    #   Add the boolean column
    df = pd.concat([df_always, df_cond], axis=1)
    df[boolean_name] = booleans

    # Reorder columns so the boolean can appear first or last, as you prefer.
    # For this example, let's put the boolean as the first column:
    cols =  always_params + [boolean_name] + cond_params
    df = df[cols]

    # -----------------
    # 6) Save and return
    # -----------------
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    # Example usage:
    always_dict = {
        'DL_Eng': [1.0, 3.0],
        'DL_EKM':  [1.0, 3.0],
        'T_Acc (MBSE)': [0.1, 0.9],
        'T_I (MBSE)':  [0.0, 1.0],
        'T_Acc (LFSystemSimulator)': [0.1, 0.9],
        'T_I (LFSystemSimulator)':  [0.0, 1.0],
        'T_Acc (CAD)': [0.1, 0.9],
        'T_I (CAD)':  [0.0, 1.0],
        'T_Acc (IDE)': [0.1, 0.9],
        'T_I (IDE)':  [0.0, 1.0],
        'T_Acc (ECAD)': [0.1, 0.9],
        'T_I (ECAD)':  [0.0, 1.0],
        'T_Acc (FEM)': [0.1, 0.9],
        'T_I (FEM)':  [0.0, 1.0],
        'T_Acc (CurcuitSimulation)': [0.1, 0.9],
        'T_I (CurcuitSimulation)':  [0.0, 1.0],
    }
    conditional_dict = {
        'T_Acc (new)': [0.1, 0.9],
        'T_I (new)': [0.0, 1.0],
        'T_Usab (new)': [1.0, 3.0]
    }
    fraction_true = 0.25
    n_samples = 724

    df_samples = create_lhs_table_with_condition(
        always_dict,
        conditional_dict,
        fraction_true,
        n_samples,
        boolean_name="New Tool",
        output_csv="latin_hypercube_DOE.csv",
        random_seed=42
    )

    print(df_samples)
