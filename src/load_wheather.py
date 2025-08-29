from main import Weather
import pandas as pd
from pathlib import Path

DATA_PATH = Path(Path(__file__).parent, "data", "Wheather")

def load_wheather(path):
    df = pd.read_csv(path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y%m%d')
    df.rename(columns={'TMED(C)': 'temp', 'RAD(MJ/M2)': 'par', 'LLUVIA(mm)': 'precip', 'EVAP_TRANS(mm)': 'et0'}, inplace=True)
    return df


df_weather = load_wheather(DATA_PATH / "weather.csv")
weather_simul = Weather(temp=df_weather['temp'], par=df_weather['par'], precip=df_weather['precip'], et0=df_weather['et0'])
