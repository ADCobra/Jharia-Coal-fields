# Land surface temperature from Landsat band 10

Single-channel retrieval: digital number to radiance (RADIANCE_MULT/ADD), radiance to brightness temperature (K1, K2), then emissivity correction from NDVI (Sobrino et al. 2004 threshold method). No atmospheric correction is applied, so a bias against the USGS Level-2 product is expected on real scenes.

## Synthetic check

The synthetic scenes are encoded with the same constants, so this mainly confirms the physics code inverts cleanly; the error left comes from noise and a lower emissivity assumed for coal.

| Scene | Compared with | Pixels | Bias (K) | MAE (K) | RMSE (K) | 95th pct abs. error (K) |
| --- | --- | --- | --- | --- | --- | --- |
| SYN_-003_20250817 | known surface temperature | 1164115 | -0.003 | 0.099 | 0.134 | 0.240 |
| SYN_-003_20250817 | synthetic Level-2 product | 1164115 | -0.003 | 0.224 | 0.283 | 0.551 |
| SYN_-002_20250902 | known surface temperature | 1168948 | -0.004 | 0.099 | 0.133 | 0.239 |
| SYN_-002_20250902 | synthetic Level-2 product | 1168948 | -0.004 | 0.224 | 0.284 | 0.552 |
| SYN_-001_20250918 | known surface temperature | 1165709 | -0.004 | 0.099 | 0.133 | 0.239 |
| SYN_-001_20250918 | synthetic Level-2 product | 1165709 | -0.003 | 0.224 | 0.283 | 0.551 |
| SYN_+000_20251004 | known surface temperature | 1168948 | -0.003 | 0.099 | 0.133 | 0.240 |
| SYN_+000_20251004 | synthetic Level-2 product | 1168948 | -0.003 | 0.224 | 0.283 | 0.552 |
| SYN_+001_20251020 | known surface temperature | 1165441 | -0.003 | 0.099 | 0.133 | 0.240 |
| SYN_+001_20251020 | synthetic Level-2 product | 1165441 | -0.003 | 0.224 | 0.283 | 0.551 |
| SYN_+002_20251105 | known surface temperature | 1166360 | -0.004 | 0.099 | 0.133 | 0.239 |
| SYN_+002_20251105 | synthetic Level-2 product | 1166360 | -0.004 | 0.224 | 0.283 | 0.551 |
| SYN_+003_20251121 | known surface temperature | 1168948 | -0.004 | 0.099 | 0.133 | 0.239 |
| SYN_+003_20251121 | synthetic Level-2 product | 1168948 | -0.003 | 0.224 | 0.283 | 0.550 |

## Real scenes

Not run here. Set FIREWATCH_EVAL_SCENES to a comma-separated list of Landsat Collection 2 Level-1 scene ids and run `firewatch -c config/landsat.yaml eval lst`.
