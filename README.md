# HazeRadar

HazeRadar is an AI-powered air quality forecasting and haze detection platform designed to monitor, predict, and visualize pollution conditions using machine learning, computer vision, and real-time environmental data.

Developed for the **YDCT Competition 2026**, the project integrates forecasting, image-based haze detection, pollution spread simulation, and environmental monitoring into a single interactive platform that helps users better understand air quality trends and potential pollution risks.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-EE4C2C?logo=pytorch)
![Deployed](https://img.shields.io/badge/Deployed-Railway-purple)

---

## Project Overview

HazeRadar combines multiple AI and data-driven technologies to provide actionable insights into environmental conditions and air pollution patterns.

The platform includes:

* PM2.5 forecasting
* Haze image detection
* Pollution spread simulation
* Real-time environmental monitoring
* Interactive visualization dashboards

---

## Features

### Air Quality Forecasting

* Predicts PM2.5 levels up to 72 hours ahead
* Uses Graph Neural Networks for spatiotemporal forecasting
* Helps anticipate pollution spikes before they occur

### Haze Detection

* Upload images for automatic haze classification
* Computer vision pipeline powered by EfficientNet-B0
* Provides instant environmental condition assessment

### Pollution Spread Simulation

* Simulates pollution movement using weather and wind data
* Visualizes potential spread patterns across regions
* Supports environmental awareness and planning

### Real-Time Monitoring

* Integrates live weather and AQI information
* Displays current environmental conditions
* Enables data-driven decision-making

### Interactive Dashboard

* User-friendly web interface
* Real-time visualizations and maps
* Easy access to forecasts and analytics

---

## Tech Stack

### Frontend

* React
* Tailwind CSS
* Leaflet
* Interactive Maps

### Backend

* Flask
* FastAPI
* Python

### AI / Machine Learning

* PyTorch
* EfficientNet-B0
* Graph Attention Networks (GATv2)

### Database & Deployment

* Supabase
* Railway
* Vercel

---

## Project Structure

```bash
HazeRadar/
│
├── frontend/           # React frontend application
├── crowdvision/        # Haze image detection module
├── gnn_forecasting/    # Air quality forecasting models
├── data_pipeline/      # Data collection and preprocessing
├── backend_api/        # Flask/FastAPI services
└── docs/               # Documentation and assets
```

## Team

Developed by:

* Mansha Khod
* Chaterina Olivia Putri Sugiarto
* Punithavathi N.C.
* Yonatan Adi Cahyoningrat

---

## Competition

HazeRadar was developed for the Youth Development for Climate Tech (YDCT) competition.

The project received a Certificate of Recognition from the organizers for demonstrating innovation, creativity, and commitment in developing impactful climate technology solutions..

---

## Future Enhancements

* Multi-city forecasting support
* Mobile application integration
* Advanced satellite imagery analysis
* Improved forecasting accuracy using additional environmental factors
* Historical trend analysis and reporting

---

## License

This project is intended for educational, research, and competition purposes.
