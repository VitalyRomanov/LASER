FROM laser_base

# Set workdir
ENV LASER /app/LASER
WORKDIR $LASER
RUN ls $LASER

COPY ./embed.sh $LASER/tasks/embed/
COPY ./decode.py $LASER/tasks/embed/decode.py

# Make port 80 available to the world outside this container
WORKDIR /app

#Open the port 80
EXPOSE 80

COPY ./app.py /app/app.py

CMD ["/bin/bash"]
