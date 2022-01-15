# sutton-barto
My take on some problems from "Reinforcement Learning: An Introduction" by Sutton &amp; Barto

__Dyna_shortcut_maze_example_8.3_reproducible__ contains everything required to build a Docker image running one experiment related to example 8.3 (for an assignment).

To run it:
1. Git clone this repository;
2. Run "docker build -t <image_name> ." in directory with Dockerfile;
3. Run "docker run -v "$(pwd)/results:/example/results" <image_name>";
4. Check out the pdf report in results.
