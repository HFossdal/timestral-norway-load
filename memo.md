# Memo: Can we use Mistral's Timestral for day-ahead load forecasts?

To: load forecasting team, Norwegian grid operator
Re: zero-shot forecasting of hourly load in NO1-NO5
Basis: tested on winter 2025-26, with forecasts made at noon the day before

## Short answer

Yes, on three conditions:

1. Give it a temperature forecast.
2. Give it 16 weeks of load history, not 4.
3. Recalibrate its uncertainty bands by temperature before you plan reserves on them.

Set up this way, it was the most accurate model I tested. With measured temperature its average error was 70 MW per zone and hour (1.9%), against 76 MW for Google's TimesFM 3.0, 77 MW for Chronos-2 and 97 MW for a standard machine-learning model (LightGBM, not tuned). With real weather forecasts, which is what you would have in practice, it was 77 MW, still the lowest. With only 4 weeks of history the differences are smaller and their significance depends on the winter and bootstrap block length; Timestral gains the most from the longer history. It needs no training on our data and runs in seconds on a laptop.

With real weather forecasts it was also more accurate than the day-ahead forecast Statnett publishes on the ENTSO-E Transparency Platform, in all four winters I could check (77 against 128 MW in winter 2025-26). Not in every zone, though: in NO1 (Oslo), Statnett's forecast was better in the three earlier winters.

## What it is good at

- Everyday accuracy: it was the best of the models tested, in every zone of the test winter.
- Daily peaks: it got the peak hour right (within an hour) on 4 days out of 5.
- Retrospective reserve-margin estimate: the zone-wise 90th percentiles of test-period daily peak errors sum to about 700 MW using real weather forecasts (600 MW with observed temperature), against 750 MW for TimesFM 3.0, 900 MW for Chronos-2 and 1,000 MW for Statnett's published forecast. These estimates use the test period's errors. They are not an out-of-sample validation of a reserve policy or a joint five-zone coverage guarantee.
- Uncertainty: with 16 weeks of history and perfect weather, its 80% band contained the actual load 80% of the time, also in cold snaps.

## Where not to trust it as it is

- With only 4 weeks of history it under-forecasts cold snaps and responds less to colder future temperatures than it does with 16 weeks. A delayed temperature effect is one possible mechanism; the synthetic experiment does not identify the cause in real demand. On days below -10 °C, the load went above its 80% band 31% of the time instead of 10%. Chronos-2 has the same one-sided misses.
- Cold that is new even to 16 weeks of history is still under-forecast, by about 30 MW per zone and hour. Watch the first deep freeze of the winter: on days below -10 °C, even the recalibrated bands are exceeded about 1 time in 5 with real weather forecasts.
- It leans on the weather forecast. With MET Norway's real day-ahead forecasts instead of measured temperature, its error rises from 70 to 77 MW, which is still better than Chronos-2 (83 MW) and TimesFM 3.0 (82 MW) with the same forecasts. In cold snaps the cost is bigger (69 to 86 MW), because the weather forecast is worst there and tends to be too mild. Its bands also become too narrow (76% coverage instead of 80%, 73% in cold snaps), since they don't account for the weather forecast being wrong.
- Over Christmas and New Year it over-forecasts load by about 40 MW, and a holiday flag doesn't help. Use a manual adjustment or last year's pattern for these days.

## How to fix the uncertainty

- Standard recalibration on its own isn't enough. It treats over- and under-forecasts alike, so it widens the band on both sides, while the cold-snap problem is one-sided: the load goes above the band and rarely below.
- Adjust the upper and lower edges separately, by temperature (below -10, -10 to -5, -5 to 0, and above 0 °C), learned from earlier winters for the main test-winter result. With observed temperatures, retrospective leave-one-winter-out checks brought cold-snap misses to approximately 10% above and 10% below for Timestral, Chronos-2 and TimesFM 3.0. Those checks can fit on later winters; they are not a deployment backtest. This is the perfect-weather setting, not the result with actual weather forecasts.
- If you have to run with a short history, a simple two-number temperature correction of the forecast also works.
- Recalibrate on forecasts made with the weather forecast you will actually use, not with observed temperatures. Tested with archived MET Nordic forecasts over four winters, this brings overall coverage back to 80%, at the cost of bands about 12% wider.
- This only partly fixes cold snaps. On cold days the weather forecast itself tends to be 1-2 °C too mild, and no adjustment based on the weather forecast can see cold it didn't predict. The load still went above the adjusted band on 14% of cold-snap hours (target 10%). Add a manual margin in cold snaps, or correct the weather forecast's cold bias first.

## Caveats

- One test winter, though the key checks were repeated over 4-5 winters.
- The context-length effect was first noticed on the test winter, then checked on the calibration and other winters. Choosing 16 weeks was therefore post hoc. The headline accuracy and temperature-response gains survive 1-, 3- and 7-day bootstrap blocks, but the intervals do not account for that selection or uncertainty over unseen winters.
- One weather station per zone. Several stations per zone lowered the error by about 5% and changed no conclusion.
- The data is metered consumption from Elhub, not the TSO's total load.
- The Statnett comparison uses its published forecast, which may differ from its operational forecast. It targets total load including losses; I scale it to Elhub consumption with the 2024-25 ratio, which postdates the two earlier comparison winters. Its nominal publication deadline is 10:00 CET the day before, while these models use load up to noon local time. The download does not establish each value's publication/revision time, and the experiment assumes real-time load measurements. This is a retrospective comparison with different information sets.
- Moirai 2.0 was tested on two winters only and could not use temperature.
- Possible contamination: Timestral was released after the test winter and its training data is not public, so it may have seen this data. Chronos-2 has not, and it shows the same patterns.
