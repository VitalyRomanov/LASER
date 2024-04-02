docker run -v $HOME/dev/LASER/models:/app/LASER/models -v $HOME/dev/LASER/source:/app/LASER/source -v $HOME/tatar/tt_parallel/ttru:/data -e PYTHONPATH='/app/LASER/' -it laser python /app/LASER/source/embed_file.py --input /data/data.tt.10 --output /data/data.tt.10.vec --encoder tat

np.fromfile("/data/data.tt.10.vec").reshape(10,-1).shape