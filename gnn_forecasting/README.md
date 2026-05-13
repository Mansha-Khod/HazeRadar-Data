# GNN Forecasting

This module handles PM2.5 forecasting using Graph Neural Networks.

The model predicts future pollution levels for connected cities using:
- weather data
- AQI data
- fire hotspot data
- spatial city relationships

## Features

- Multi-city forecasting
- Graph-based prediction
- 72-hour PM2.5 forecasting
- Real-time inference pipeline

## Model

- Graph Attention Network (GATv2)

## Inputs

- Temperature
- Humidity
- Wind speed
- Wind direction
- Fire hotspot density
- Current PM2.5

## Output

Predicted PM2.5 values for future timestamps.

## Tech Stack

- PyTorch
- PyTorch Geometric
- FastAPI
- Supabase
