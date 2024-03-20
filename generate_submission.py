import pandas as pd
from catboost import CatBoostRegressor
from do_eda import prepare_data

data_to_subm = prepare_data()

cb_regr = CatBoostRegressor()
cb_regr.load_model('classifier')
S_hat = cb_regr.predict(data_to_subm.drop(columns='id'))

submit = pd.DataFrame(data_to_subm['id'])
submit['score'] = S_hat
submit.to_csv('output/submission.csv', index=None)

print("\nSubmit done!")



