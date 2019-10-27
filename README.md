# XAircraft

This repository provides the simulation models of aircraft.
This is including the a body dynamics model, a control rudder dynamics.
The non-linear aircraft dynamics model shall be implemented in the future.

The models are provided as 2 types, numpy-based and tensorflow2-based.
Tensorflow1.x is not supported.

## Requirements
- python > 3.7
- tensorflow > 2.0
- numpy > 1.17.2
- xtools > 0.1

For the information of xtools, see [here](https://github.com/xikasan/xtools).

# Models
 - LVAircraft
    - name : `lvaircraft-v0`
    - Longitudinal dynamics of aircraft body.
 - LHAircraft
    - name : `lhaircraft-v0`
    - Lateral dynamics of aircraft body.