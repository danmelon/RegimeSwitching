import os
import data_pipeline, hmm_model, ms_model, regime_evaluate

os.makedirs("outputs", exist_ok=True)

train_scaled, test_scaled, train, test, prices, scaler = data_pipeline.run()
regimes, hmm, hmm_bull, hmm_bear = hmm_model.run(train_scaled, test_scaled, train, test, prices, scaler)
ms_result, ms_bull, ms_bear      = ms_model.run(train, test, prices)
regime_evaluate.run(train, test, regimes, ms_result, hmm_bull, ms_bull, ms_bear, prices)