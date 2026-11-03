# Memo: Can we use Mistral's Timestral for day-ahead load forecasts?

To: load forecasting team, Norwegian grid operator
Re: zero-shot forecasting of hourly load in NO1-NO5
Basis: tested on winter 2025-26, with forecasts made at noon the day before

## Short answer

Yes, on three conditions:

1. Give it a temperature forecast.
2. Give it 16 weeks of load history, not 4.
3. Recalibrate its uncertainty bands by temperature before you plan reserves on them.

Set up this way, it was the most accurate model I tested. With measured temperature its average error was 70 MW per zone and hour (1.9%), against 76 MW for Google's TimesFM 3.0, 77 MW for Chronos-2 and 97 MW for a standard machine-learning model (LightGBM, not tuned). With real weather forecasts, which is what you would have in practice, it was 77 MW, still the lowest. With only 4 weeks of history these three foundation models are about equally accurate; Timestral gains the most from the longer history. It needs no training on our data and runs in seconds on a laptop.

With real weather forecasts it was also more accurate than the day-ahead forecast Statnett publishes on the ENTSO-E Transparency Platform, in all four winters I could check (77 against 128 MW in winter 2025-26). Not in every zone, though: in NO1 (Oslo), Statnett's forecast was better in the three earlier winters.

## What it is good at

- Everyday accuracy: it was the best of the models tested, in every zone of the test winter.
- Daily peaks: it got the peak hour right (within an hour) on 4 days out of 5.
- Reserve margin: to cover the real daily peak on 9 days out of 10, you would hold about 700 MW above its forecast, summed over the five zones, using real weather forecasts (600 MW with perfect weather). With the same weather forecasts, TimesFM 3.0 needs about 750 MW, Chronos-2 about 900 MW, and Statnett's published forecast about 1,000 MW.
- Uncertainty: with 16 weeks of history and perfect weather, its 80% band contained the actual load 80% of the time, also in cold snaps.

## Where not to trust it as it is

- With only 4 weeks of history it under-forecasts cold snaps. It learns how strongly cold drives load from the history you give it, and four weeks of mild autumn teach it too little. On days below -10 °C, the load went above its 80% band 31% of the time instead of 10%. Chronos-2 has the same weakness.
- Cold that is new even to 16 weeks of history is still under-forecast, by about 30 MW per zone and hour. Watch the first deep freeze of the winter: on days below -10 °C, even the recalibrated bands are exceeded about 1 time in 5 with real weather forecasts.
- It leans on the weather forecast. With MET Norway's real day-ahead forecasts instead of measured temperature, its error rises from 70 to 77 MW, which is still better than Chronos-2 (83 MW) and TimesFM 3.0 (82 MW) with the same forecasts. In cold snaps the cost is bigger (69 to 86 MW), because the weather forecast is worst there and tends to be too mild. Its bands also become too narrow (76% coverage instead of 80%, 73% in cold snaps), since they don't account for the weather forecast being wrong.
- Over Christmas and New Year it over-forecasts load by about 40 MW, and a holiday flag doesn't help. Use a manual adjustment or last year's pattern for these days.

## How to fix the uncertainty

- Standard recalibration on its own isn't enough. It treats over- and under-forecasts alike, so it widens the band on both sides, while the cold-snap problem is one-sided: the load goes above the band and rarely below.
- Use a one-sided recalibration instead. Adjust the upper and lower edges separately, by forecast temperature (for example below -10, -10 to -5, -5 to 0, and above 0 °C), learned from earlier winters. In my tests this brought the cold-snap misses back to about 10% above and 10% below, for Timestral, Chronos-2 and TimesFM 3.0.
- If you have to run with a short history, a simple two-number temperature correction of the forecast also works.
- Recalibrate on forecasts made with the weather forecast you will actually use, not with observed temperatures. Tested with archived MET Nordic forecasts over four winters, this brings overall coverage back to 80%, at the cost of bands about 12% wider.
- This only partly fixes cold snaps. On cold days the weather forecast itself tends to be 1-2 °C too mild, and no adjustment based on the weather forecast can see cold it didn't predict. The load still went above the adjusted band on 14% of cold-snap hours (target 10%). Add a manual margin in cold snaps, or correct the weather forecast's cold bias first.

## Caveats

- One test winter, though the key checks were repeated over 4-5 winters.
- One weather station per zone. Several stations per zone lowered the error by about 5% and changed no conclusion.
- The data is metered consumption from Elhub, not the TSO's total load.
- The Statnett comparison uses the forecast it publishes on the Transparency Platform, which may not be the one it plans with. My models also assume the load is measured in real time up to noon.
- Moirai 2.0 was tested on two winters only and could not use temperature.
- Possible contamination: Timestral was released after the test winter and its training data is not public, so it may have seen this data. Chronos-2 has not, and it shows the same patterns.
