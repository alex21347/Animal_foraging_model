# Animal_foraging_model
Modelling the movements of foraging animal as a Levy Flight trained via evolutionary algorithm <br/>
<br/>
The code in this repo is somewhat chronological. <br/>
<br/>
**levy_flight.py** is the code that simply builds a Levy flight in 2 dimensions (R^2). <br/>
**levy_flight_foraging.py** takes the levy flight and simulates the concept of an animal foraging/searching for food. <br/>
**levy flight ml.py** applies an evolutionary algorithm (stochastic gradient descent) on the parameter space that governs the Levy flight
and thus providing evidence that over time animals have evolved from searching randomly, to utilising the Levy flight as a means of
efficienty searching for food. https://en.wikipedia.org/wiki/L%C3%A9vy_flight_foraging_hypothesis <br/>

![Animal Foraging Example](foraging_plots.png)
 
