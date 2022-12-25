

def get_num_vars_lists(data_schema):      
    num_vars = []
    attributes = data_schema["inputDatasets"]["clusteringBaseMainInput"]["inputFields"]   
    for attribute in attributes: 
        num_vars.append(attribute["fieldName"])
    return num_vars 


def verify_data_columns_in_schema(data, pp_params): 
    useable_vars = [var for var in pp_params["num_vars"] if var in data.columns]    
    if len(useable_vars) == 0:
        raise Exception('''
            Error: Given training data does not have any input attributes expected as per 
            the input schema. Do you have the wrong data, or the wrong schema? ''')
    return 


def get_preprocess_params(data, data_schema, model_cfg): 
    # initiate the pp_params dict
    pp_params = {}   
            
    # set the id attribute
    pp_params["id_field"] = data_schema["inputDatasets"]["clusteringBaseMainInput"]["idField"]   
    
    # get the list of categorical and numeric variables and set in the params dict
    num_vars = get_num_vars_lists(data_schema)    
    pp_params["num_vars"] = num_vars       
    
    # create list of variables to retain in the data - id, cat_vars, and num_vars
    pp_params["retained_vars"] = [pp_params["id_field"]] + num_vars    
    
    # verify that the given data matches the input_schema
    verify_data_columns_in_schema(data, pp_params)   
    
    # pprint.pprint(pp_params)    
    return pp_params


