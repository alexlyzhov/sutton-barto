# Base image
FROM python:3.7

# Author information
MAINTAINER nikkou.postbox@gmail.com

# Set a working directory (new dir!)
WORKDIR example

# Install latex.
RUN apt-get update && apt-get install -y texlive

# Install necessary libraries
# pip install -r requirements.txt
RUN pip install numpy matplotlib

# Add necessary files. Good practice to do it at the end
# in order to avoid reinstallation of dependencies when files change
ADD code ./code
ADD latex ./latex
ADD run.sh ./

# Make run.sh executable
RUN chmod +x run.sh

# /example/results contents will be shared with the host
# if -v option used with "docker run" command
VOLUME /example/results

CMD ./run.sh