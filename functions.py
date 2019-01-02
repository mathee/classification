""" needs functions for:
    EXPLORATION
    + correlation matrix
    
    corr = sandbox.corr()
    pyplot.figure(figsize=(15, 8))
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True)
    
    + show mean classes for specific grouped categories
    
    FEATURE ENGINEERING
    + one hot encoding: creating columns for each code, automatic naming
    + scaling 
    + further binning? a function that creates bins depending on another function, mapping, ranges etc
    + filling empty rows, gaps etc
    + slicing between first and last valid row?
    + everything else should be done individually maybe...
    

    
    
    
"""