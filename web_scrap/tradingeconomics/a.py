import pandas as pd
import tradingeconomics as te
te.login('guest:guest')

## Without a client key only a small sample of data will be given.

## Putting country name or indicator name in square brackets [] will result, by default
## in the dictionary type for several countries and indicators.
## EXE: country=['mexico', 'sweden']

## With no output_type defined, the result will be of the dictionary type.
## Use output_type='df' to display in pandas dataframe. 

# To get historical data by specific country, indicator and start date
mydata = te.getHistoricalData(country='mexico', indicator='gdp', initDate='2009-01-01', output_type='df')
print(mydata)
print("===============================================================================================================")