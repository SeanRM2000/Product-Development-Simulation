import numpy as np
import pandas as pd
from scipy.stats import qmc

def create_lhs_table_with_condition(
    values,
    fraction_true,
    n_samples,
    boolean_name="my_boolean",
    output_csv="lhs_samples.csv",
    random_seed=None
):

    # -----------------
    # 1) Prep Samplers
    # -----------------
    if random_seed is not None:
        np.random.seed(random_seed)

    params = list(values.keys())
    lower_always = np.array([values[p][0] for p in params])
    upper_always = np.array([values[p][1] for p in params])


    # Create LHS samplers for each set
    sampler_always = qmc.LatinHypercube(d=len(params), seed=random_seed)

    # -----------------
    # 2) Generate LHS samples
    # -----------------
    raw_always = sampler_always.random(n_samples)  # shape (n_samples, len(always_params))


    # Scale them to the actual ranges
    scaled = qmc.scale(raw_always, lower_always, upper_always)

    # -----------------
    # 3) Generate Boolean column
    # -----------------
    # fraction_true of them should be True. We can do this by random uniform draws:
    booleans = np.random.rand(n_samples) < fraction_true
    # booleans is an array of shape (n_samples,), with True in ~ fraction_true portion.

    # -----------------
    # 4) Build the DataFrame
    # -----------------
    df = pd.DataFrame(scaled, columns=params)


    #   Add the boolean column
    df[boolean_name] = booleans

    # Reorder columns so the boolean can appear first or last, as you prefer.
    # For this example, let's put the boolean as the first column:
    cols =  params + [boolean_name]
    df = df[cols]

    # -----------------
    # 6) Save and return
    # -----------------
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    # Example usage:
    values = {
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
        'T_Acc (new)': [0.1, 0.9],
        'T_I (new)': [0.0, 1.0],
        'T_Usab (new)': [1.0, 3.0]
    }
    fraction_true = 0.25
    n_samples = 724

    df_samples = create_lhs_table_with_condition(
        values,
        fraction_true,
        n_samples,
        boolean_name="New Tool",
        output_csv="latin_hypercube_DOE.csv",
        random_seed=42
    )

    print(df_samples)
