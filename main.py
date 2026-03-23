# main.py
import data_pipeline, hmm_model

train_scaled, test_scaled, train, test, prices, scaler = data_pipeline.run()
regimes, model = hmm_model.run(train_scaled, test_scaled, train, test, prices, scaler)