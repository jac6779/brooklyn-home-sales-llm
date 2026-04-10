# Brooklyn Home Sales — Unique Categorical Values

This reference section documents the main categorical values used in the Brooklyn home sales project. It is designed for inclusion in a GitHub `README.md` so reviewers can quickly understand the geographic and property-type coverage of the dataset.

## Dataset Fields Covered

- `neighborhood`
- `zip_code`
- `building_class_category` (presented below as **Building Type**)

## Neighborhood (54 unique values)

- `bath_beach`
- `bay_ridge`
- `bedford_stuyvesant`
- `bensonhurst`
- `bergen_beach`
- `boerum_hill`
- `borough_park`
- `brighton_beach`
- `brooklyn_heights`
- `brownsville`
- `bushwick`
- `canarsie`
- `carroll_gardens`
- `clinton_hill`
- `cobble_hill`
- `cobble_hill_west`
- `coney_island`
- `crown_heights`
- `cypress_hills`
- `dyker_heights`
- `east_new_york`
- `flatbush_central`
- `flatbush_east`
- `flatbush_lefferts_garden`
- `flatbush_north`
- `flatlands`
- `fort_greene`
- `gerritsen_beach`
- `gowanus`
- `gravesend`
- `greenpoint`
- `kensington`
- `madison`
- `manhattan_beach`
- `marine_park`
- `midwood`
- `mill_basin`
- `ocean_hill`
- `ocean_parkway_north`
- `ocean_parkway_south`
- `old_mill_basin`
- `park_slope`
- `park_slope_south`
- `prospect_heights`
- `red_hook`
- `seagate`
- `sheepshead_bay`
- `sunset_park`
- `williamsburg_central`
- `williamsburg_east`
- `williamsburg_north`
- `williamsburg_south`
- `windsor_terrace`
- `wyckoff_heights`


## Building Type (5 unique values)

These values come from the `building_class_category` field.

- `one_family_dwellings`
- `rentals_elevator_apartments`
- `rentals_walkup_apartments`
- `three_family_dwellings`
- `two_family_dwellings`

## Notes

- Neighborhood names are shown in the cleaned format used in the modeling dataset.
- ZIP Codes are shown as numeric Brooklyn ZIPs represented in the analysis file.
- Building Type values reflect the residential categories retained for exploratory analysis and modeling.
