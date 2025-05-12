our = df.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)