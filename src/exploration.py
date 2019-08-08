import pandas as pd

if __name__ == "__main__":
    complaints = pd.read_csv('/Users/ak/PycharmProjects/comp_embed/data/raw/case_study_data.csv')
    print('Data set has {n} records'.format(n=len(complaints)))
    distinct_ids = list(set(complaints.complaint_id.tolist()))
    print('Data set has {i} distinct complaint IDs'.format(i=len(distinct_ids)))
    grouped_by_prod = complaints.groupby(['product_group']).count()
    print(grouped_by_prod)
    complaints['text_len'] = complaints['text'].apply(lambda x: len(x))
    print(complaints.head())
    text_len_by_prod = complaints.groupby(['product_group']).agg({'text_len':
                                                                      {'min_len': 'min',
                                                                       'mean_len': 'mean',
                                                                       'median_len': 'median',
                                                                       'max_len': 'max',
                                                                       'sdev': 'std',
                                                                       'dist_skew': 'skew'}})
    print(text_len_by_prod)
