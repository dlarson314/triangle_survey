# triangle_survey
Do survey computations for a measured triangular grid

As an example, we consider a loop of triangles.  We start at points A and B, and
measure triangles until we get back to those points.  In the survey file, we
label the points A_close and B_close when returning to them, so that we can
determine the level of error in the survey.  In the figure below, I use 100 foot
equilateral triangles to make the loop, and give each of the edges 3% error in
length measurements.
![Uncorrected survey](survey2_uncorrected.png)

Then, I require A and A_close to be the same point, and I require B and B_close
to be the same point, and I use scipy.optimize.leastsq to find a least squares
solution for all the point locations.  This gives the following result:
![Uncorrected survey](survey2_both.png)

