# READ ME
## BANG: BANG Aerodynamic Numerical Gigasim

## What is BANG?
BANG is a fluid-dynamics simulation made for a school project. It can simulate the flow of air around a wing and show the density, the internal energy and the speed in clear graphs.

## How to download
Find the github release page for BANG and download the correct BANG file (BANG.exe for windows and for linux BANG). After downloading the correct file launch the file and you'll be put straight into the simulation.

## How to operate BANG
![Screenshot of the simulation](Afbeelding1.png)

In the above screenshot you can see our GUI(Graphics User Interface). You'll be able to see two sliders, one at the top and one at the bottom. The top slider functions as a rotation device for the wing, you can turn it from -180 degrees to +180 degrees. Everytime you change any parameter of the simulation it will restart. The lower slider is for the speed, it starts at 343 meters per second i.e. the speed of sound. You can change it from 0 meters per second all the way to 500 meters per second. In the middle you can see the four graphs depicting the density of the air in kg/m^3, the internal energy in J/kg, the speed in m/s and the speed in mach. Each colourscale adjusts automatically based on the maximum and minimum values currently depicted on the graph. Below all that you'll find seven buttons, the rightmost labeled 'random' will turn your wing into any random NACA wing. Then there are five wing presets from left to right: NACA0012: a symmetrical wing, NACA2410: a somewhat cambered wing, NACA6412: a highly cambered wing used for STOL aircraft (Short TakeOff and Landing), NACA9402: an extremely cambered and super thin wing that woudl never be used in real life but is there to show extremities: NACA6338, as with the last wing this one is again highly unrealistic, it's very thick and has a high camber. All the way on the right you'll see a button labeled 'reset', it simply resets the simulation like changing any parameter would also. Just above the buttons you can see a set of numbers, the first one shows the calculated total drag on the wing, the second the calculated lift of the wing. 