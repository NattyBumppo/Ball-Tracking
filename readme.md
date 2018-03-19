Ball-Tracking
===============

This is a Python application that uses the OpenCV library (version 2) to track the movement of balls during a video. It tracks the balls with a combination of color filtering and k-means clustering. Once the balls' positions are determined, their velocities are estimated by comparing consecutive frames. Then numerical integration (Euler's method) is used to find a predicted trajectory through space (subject to gravity). Ultimately, these trajectories are updated with new data from the image processing side, using a weighted filter to combine the predicted trajectory with the observed position and velocity.

![Ball tracking](https://cloud.githubusercontent.com/assets/118711/7895258/35512a12-0637-11e5-946f-b4c86664c3c3.png "Ball tracking")
