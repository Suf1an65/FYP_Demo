# This is impossible to run indepndently without docker image to run openface for feature extraction. 
To run the frontend use command npm run dev.
to run the backend use the command. uvicorn main:app --reload --host 127.0.0.1 --port 8000 
That said do not run this unless you have a custome docker image that allows you to run poenface as well as ffmpeg at the system level.
