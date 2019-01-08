""" script contains problem-specifc feature engineering steps/functions that 
create new features - going beyond standardized, automated preprocessing steps 
like scaling/encoding/deleting. --> Everything that requires domain knowledge.
INPUT: DataFrame
OUTPUT: DataFrame
"""

###############################################################################
# GENERAL FUNCTIONS

# FIND AND CREATE INTERSECTION TERMS

def create_shifted_columns(df, column_name, min_shift, max_shift, shift_stepsize, prefix):
    '''For timeseries: Creates shifted columns for [column_name], in range of min-max-step,
    naming them "prefix(column_name-shiftvalue)"'''
    temp = column_name
    for i in range(min_shift ,max_shift + 1, shift_stepsize):
        df[f"{prefix}({temp})-{i}"] = df[[f"{temp}"]].shift(i)
        
def create_polynomials(df, column_name, min_power, max_power, stepsize, prefix):
    '''Creates polynomial columns for [column_name] in range of min-max-step,
    naming them "prefix(column_name**power)"'''
    powers = list(range(min_power, max_power+1, stepsize))
    for i in powers:
        df[f"{prefix}({column_name})**{i}"] = df[column_name] ** i

###############################################################################
# MAIN FUNCTION
def engineer_train(X):
    print(f"AFTER FEATURE ENGINEERING: {type(X)} - {X.shape}\n")
    return X

def engineer_test(X):
    print(f"AFTER FEATURE ENGINEERING: {type(X)} - {X.shape}\n")
    return X