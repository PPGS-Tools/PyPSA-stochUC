import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
src = os.path.dirname(current) 
sys.path.append(src) 
data = os.path.join(os.path.dirname(src),"data")

from forecast import Forecast

FORECAST_NAMES = ["da","aFRR_pos","aFRR_neg","q_fern"]
PRICE_COLUMNS = ['aFRR_neg_EUR_MW', 'aFRR_pos_EUR_MW', 'da_EUR_MWh']

def loadForecasts()->dict[str,Forecast]:
    forecasts: dict[str,Forecast] = {}
    from util import testData
    df = testData()
    for forecastName in FORECAST_NAMES:
        pickleFile = os.path.join(data,f"models/{forecastName}.pickle")
        try:
            forecast = Forecast.load(pickleFile)
        except:
            raise Exception("Could not load forcast model")
        if forecast.y is None:
            df2 = df.asfreq(forecast.freq)
            # If forecast does not contain data, load data
            y_name = forecast.y_name
            X_columns = forecast.X_columns
            forecast.y = df2[y_name]
            forecast.X = df2[X_columns]
        forecasts[forecastName]=forecast
    return forecasts

def saveForecasts(forecasts:dict[str,Forecast],remove_data: bool = True):
    for forecastName in FORECAST_NAMES:
        forecast = forecasts[forecastName]
        pickleFile = os.path.join(data,f"models/{forecastName}.pickle")
        forecast.save(pickleFile,remove_data=remove_data)

def fitForecasts()->dict[str,Forecast]:
    forecasts: dict[str,Forecast] = {}
    print(f"Fitting {len(FORECAST_NAMES)} forecasts ...")
    print("\t⏱️  Loading test data")
    from util import testData
    df = testData()

    # fit forecast models
    print("\t🏗️  Fitting da model")
    forecasts["da"] = Forecast(
        y=df["da_EUR_MWh"],
        X=df[["residual_load_MW"]],
        arima_order=(1,1,4),
        seasonalitiesList=[8,12,24,168,8760],
        price_columns=PRICE_COLUMNS,
        lowTol=True)
    
    print("\t🏗️  Fitting aFRR_pos model")
    forecasts["aFRR_pos"] = Forecast(
        y=df["aFRR_pos_EUR_MW"],
        X=df[["residual_load_MW","Weekend"]],
        arima_order=(5,1,2),
        seasonalitiesList=[3,6,42,2190],
        price_columns=PRICE_COLUMNS,
        lowTol=False,
        freq="4h")
    
    print("\t🏗️  Fitting aFRR_neg model")
    forecasts["aFRR_neg"] = Forecast(
        y=df["aFRR_neg_EUR_MW"],
        X=df[["residual_load_MW","Weekend"]],
        arima_order=(5,1,5),
        seasonalitiesList=[3,6,42,2190],
        price_columns=PRICE_COLUMNS,
        lowTol=False,
        freq="4h")
    
    print("\t🏗️  Fitting q_fern model")
    forecasts["q_fern"] = Forecast(
        y=df["q_fern_MW"],
        X=df[["temp_C"]],
        arima_order=(2,1,1),
        seasonalitiesList=[12,24,8760],
        price_columns=PRICE_COLUMNS,
        lowTol=False)
    
    print("✅  All forecasts fitted")
    print("💾  Saving forecasts...")
    saveForecasts(forecasts,remove_data=True)
    print("✅  All forecasts saved")
    return forecasts
if __name__ == "__main__":
    fitForecasts()