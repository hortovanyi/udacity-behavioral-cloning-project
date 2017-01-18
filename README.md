Clone driving behaviour using Deep Learning
===


With this behaviour cloning project, we give steering & throttle instruction to a vehicle in a simulator based on receiving a centre camera image and telemetry data. The steering angle data is a prediction for a neural network model trained against data saved from track runs I performed.
![simulator screen sot](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/Self_Driving_Car_Nanodegree_Program.png)

The training of the neural net model, is achieved with driving behaviour data captured, in training mode, within the simulator itself. Additional preprocessing occurs as part of batch generation of data for the neural net training.

##Model Architecture

I decided to as closely as possible use the [Nvidia's End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) model. I diverged by passing cropped camera images as RGB, and not YUV, with adjusting brightness and by using the steering angle as is. I experimented with using 1/r (inverse turning radius) as input but found the values were too small (I also did not know the steering ratio and wheel base of the vehicle in the simulator).

Additional experimentation occurred with using [comma.ai](http://comma.ai/), [Steering angle prediction model](https://github.com/commaai/research/blob/master/train_steering_model.py) but the number of parameters was higher then the nvidia model and it worked off of full sized camera images. As training time was significantly higher, and initial iterations created an interesting off road driving experience in the simulator, I discontinued these endeavours.

The model represented here is my implementation of the nvidia model mentioned previously. It is coded in python using keras (with tensor flow) in model.py and returned from the build_nvidia_model method. The complete project is on github here [Udacity Behaviour Cloning Project](https://github.com/hortovanyi/udacity-behavioral-cloning-project)

###Input

The input is 66x200xC with C = 3 RGB color channels.

###Architecture
**Layer 0: Normalisation** to range -1, 1 (1./127.5 -1)

**Layer 1: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 31x98x24, with **elu activation** and **dropout**

**Layer 2: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 14x47x36, with **elu activation** and **dropout**

**Layer 3: Convolution** with strides=(2,2), valid padding, kernel 5x5 and output shape 5x22x48, with **elu activation** and **dropout**

**Layer 4: Convolution** with strides=(1,1), valid padding, kernel 3x3 and output shape 3x20x64, with **elu activation** and **dropout**

**Layer 5: Convolution** with strides=(1,1), valid padding, kernel 3x3 and output shape 1x18x64, with **elu activation** and **dropout**

**flatten** 1152 output

**Layer 6: Fully Connected** with 100 outputs and **dropout**

**Layer 7: Fully Connected** with 50 outputs and **dropout**

**Layer 8: Fully Connected** with 10 outputs and **dropout**

dropout was set aggressively on each layer at .25 to avoid overtraining
###Output

**Layer Fully Connected** with 1 output value for the steering angle.

###Visualisation
[Keras output plot (not the nicest visuals)](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/model.png)

##Data preprocessing and Augmentation
The simulator captures data into a csv log file which references left, centre and right captured images within a sub directory. Telemetry data for steering, throttle, brake and speed is also contained in the log. Only steering was used in this project.

My initial investigation and analysis was performed in a Jupyter Notebook [here](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/behavorial-cloning-initial-data-exploration.ipynb).

Before being fed into the model, the images are cropped to 66x200 starting at height 60 with width centered - [A sample video of a run cropped](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_run1.mp4?raw=true).

![Cropped left, centre and right camera image](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/3cameras.png)

As seen in the following histogram a significant proportion of the data is for driving straight and its lopsided to left turns (being a negative steering angle is left) when using data generated following my conservative driving laps.
![Steering Angle Histogram](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/steering_histogram.png)

The log file was preprocessed to remove contiguous rows with a history of >5 records, with a 0.0 steering angle. This was the only preprocessing done outside of the batch generators used in training (random rows are augmented/jittered for each batch at model training time).

A left, centre or right camera was selected randomly for each row, with .25 angle (+ for left and - for right) applied to the steering.

Jittering was applied per [Vivek Yadav's post ](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0) to augment data. Images were randomly transformed in the x range by 100 pixels and in the y range by 10 pixels with 0.4 per xpixel adjusted against the steering angle. Brightness via a HSV (V channel) transform (.25 + a random number in range 0 to 1) was also performed.
![jittered image](https://raw.githubusercontent.com/hortovanyi/udacity-behavioral-cloning-project/master/images/jittered_center_camera.png)

During batch generation, to compensate for the left turning, 50% of images were flipped (including reversing steering angle) if the absolute steering angle was > .1.

Finally images are cropped per above before being batched.

###Model Training

Data was captured from the simulator. I drove conservatively around the track three times paying particular attention to the sharp right turn. I found connecting a PS3 controller allowed finer control then using the keyboard. At least once I waited till the last moment before taking the turn. This seems to have stopped the car ending up in the lake. Its also helped to overcome a symptom of the bias in the training data towards left turns. To further offset this risk, I validated the training using a test set I'd captured from the second track, which is a lot more windy.

####[Training sample captured of left, centre and right cameras cropped](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_run1.mp4?raw=true)
<video width="960" height="150" controls>
  <source src="https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_run1.mp4?raw=true">
</video>

Center camera has the steering angle and 1/r values displayed.

####[Validation sample captured of left, centre and right cameras cropped](https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_runt2.mp4?raw=true)
<video width="960" height="150" controls>
  <source src="https://github.com/hortovanyi/udacity-behavioral-cloning-project/blob/master/simulator_runt2.mp4?raw=true">
</video>
Center camera has the steering angle and 1/r values displayed.

The Adam Optimizer was used with a mean squared error loss. A number of hyper-parameters were passed on the command line. The command I used looks such for a batch size of 500, 10 epochs (dropped out early if loss wasn't improving), dropout at .25 with a training size of 50000 randomly augmented features with adjusted labels and 2000 random features & labels used for validation

```
python model.py --batch_size=500 --training_log_path=./data --validation_log_path=./datat2 --epochs 10 \
--training_size 50000 --validation_size 2000 --dropout .25
```


###Model Testing
To meet requirements, and hence pass the assignment, the vehicle has to drive around the first track staying on the road and not going up on the curb.

The model trained (which is saved), is used again in testing. The simulator feeds you the centre camera image, along with steering and throttle telemetry. In response you have to return the new steering angle and throttle values. I hard coded the throttle to .35. The image was cropped, the same as for training, then fed into the model for prediction giving the steering angle.

```python

steering_angle = float(model.predict(transformed_image_array, batch_size=1))
throttle = 0.35
```

####Successful run track 1
[![Successful run track 1](http://img.youtube.com/vi/aLrV8UMqzKo/0.jpg)](http://www.youtube.com/watch?v=aLrV8UMqzKo)
####Successful run track 2
[![Successful run track 2](http://img.youtube.com/vi/sW2D1T3ev-k/0.jpg)](http://www.youtube.com/watch?v=sW2D1T3ev-k)

note: the trained model I used for the track 1 run, is different to the one used to run the simulator in track 2. I found that the data I originally used to train a model to run both tracks, would occasionally meander on track 1 quite wildly. Thus used training data to make it more conservative to meet requirements for the projects.
