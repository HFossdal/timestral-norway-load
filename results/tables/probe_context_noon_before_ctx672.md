|    | model          | spec                          | term         |   coef |     lo |    hi |
|---:|:---------------|:------------------------------|:-------------|-------:|-------:|------:|
|  0 | chronos2       | w16                           | w16          |  0.003 | -0.01  | 0.019 |
|  1 | chronos2       | w16 + window_range            | w16          |  0.039 | -0.003 | 0.076 |
|  2 | chronos2       | w16 + window_range            | window_range | -0.004 | -0.007 | 0     |
|  3 | chronos2       | w16 + window_range + day_temp | w16          |  0.022 | -0.018 | 0.062 |
|  4 | chronos2       | w16 + window_range + day_temp | window_range | -0.002 | -0.006 | 0.002 |
|  5 | chronos2       | w16 + window_range + day_temp | day_temp     |  0.003 |  0     | 0.006 |
|  6 | timesfm3       | w16                           | w16          |  0.123 |  0.108 | 0.14  |
|  7 | timesfm3       | w16 + window_range            | w16          |  0.101 |  0.056 | 0.143 |
|  8 | timesfm3       | w16 + window_range            | window_range |  0.002 | -0.002 | 0.007 |
|  9 | timesfm3       | w16 + window_range + day_temp | w16          |  0.088 |  0.045 | 0.13  |
| 10 | timesfm3       | w16 + window_range + day_temp | window_range |  0.004 | -0.001 | 0.008 |
| 11 | timesfm3       | w16 + window_range + day_temp | day_temp     |  0.002 | -0.001 | 0.006 |
| 12 | timestral-209m | w16                           | w16          |  0.269 |  0.257 | 0.28  |
| 13 | timestral-209m | w16 + window_range            | w16          |  0.225 |  0.186 | 0.26  |
| 14 | timestral-209m | w16 + window_range            | window_range |  0.005 |  0.001 | 0.008 |
| 15 | timestral-209m | w16 + window_range + day_temp | w16          |  0.194 |  0.159 | 0.227 |
| 16 | timestral-209m | w16 + window_range + day_temp | window_range |  0.008 |  0.005 | 0.011 |
| 17 | timestral-209m | w16 + window_range + day_temp | day_temp     |  0.006 |  0.003 | 0.009 |