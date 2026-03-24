# main.py
import data_pipeline, hmm_model, ms_model

train_scaled, test_scaled, train, test, prices, scaler = data_pipeline.run()
regimes, model = hmm_model.run(train_scaled, test_scaled, train, test, prices, scaler)
ms_result, bull_state, bear_state = ms_model.run(train, test, prices)