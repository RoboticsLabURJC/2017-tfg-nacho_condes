
# Final Degree Project (Nacho Condés)
__Deep Learning Applications for Robotics using TensorFlow and JdeRobot__


[Full report here](https://gsyc.urjc.es/jmplaza/students/tfg-deep_learning-object_detector-nacho_condes-2018.pdf)

[Project MediaWiki here](http://jderobot.org/Naxvm-tfg)

## Requirements
* You will need to install JdeRobot for this component to work (preferably from source), following the [installation guide](https://jderobot.org/Installation#From_source_code_at_GitHub).
* All the necessary Python packages have been annotated for <code>pip</code> to install them automatically. To do so, run:
`pip2 install -r requirements.txt`
* (Recommended) [Install TensorFlow from sources](https://www.tensorflow.org/install/install_sources) (much more efficient than the generic version installed above via `pip`). If you are equipped with an Nvidia GPU, use it, as it is way faster than the CPU version.
* Install the required ROS packages, to handle the cameras information.
    * PTZ: `sudo apt install ros-kinetic-usb-cam`
    * Turtlebot2: `sudo apt install ros-kinetic-openni2_launch ros-kinetic-kobuki-node`


_For PTZ camera_:
Make sure that you execute `sudo chmod 666 /dev/ttyUSB0` when you connect the PT motors (EVI connector) to your computer (`evicam_driver` needs this to be this way, otherwise it will raise an _EBADF_ error when trying to access the device).

Also, check which of your computer video devices corresponds to the PT camera interface. You can perform this launching `ls /dev`. You will see the devices related to your computer. `/dev/video0` is tipically your laptop webcam (or default camera). The PT camera will correspond to the next device, which can stand for `/dev/video1`, `/dev/video2`, etc. This is due to the order of the USB connections. You will have to change the value of the `resources/usb_cam-test.launch` file to match to this device no (line 2):
  `<param name="video_device" value="/dev/your_video_device" />`

***
## FollowPerson

Application capable of implementing a robotic behavioral to follow a determined person (__mom__), commanding movements to a robot (Turtlebot2) or a PTZ Camera (Sony EVI D100P). It uses Deep Learning to do so: a detection CNN (_SSD_ Architecture), plus a face reidentification CNN (_FaceNet_ architecture), both of them implemented on TensorFlow.

The implementation (network models and mom image) can be customized using the YML file (`turtlebot.yml` or `ptz.yml`)


**Functional video:**


[![YouTube video](http://img.youtube.com/vi/oKMR_QCT7EE/0.jpg)](https://www.youtube.com/watch?v=oKMR_QCT7EE)
<!--[![YouTube video](http://img.youtube.com/vi/ZH4MJVXKo1w/0.jpg)](https://www.youtube.com/watch?v=ZH4MJVXKo1w) -->


### How to use

**0. Tune your execution**

* Object Detection model: you can download a pre-trained network model from the [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Choose among those which output _boxes_ (not regions). Just download the .zip and keep the `.pb` file (which contains the frozen graph structure and weights). Place it into the `Net/TensorFlow` directory, and indicate its name in the suitable YML file (in the `FollowPerson.Network.Model` node). In addition, you will have to indicate in the `FollowPerson.Network.Dataset` node which was the training dataset of that model (you can check it in the Model Zoo page).

* FaceNet model: you can download a TensorFlow model from [this FaceNet implementation](https://github.com/davidsandberg/facenet#pre-trained-models). Extract the .zip file and place the `.pb` file inside the `Net/TensorFlow` directory. Indicate the file name in the YML configuration file you wish to use (depending on the device), in the `FollowPerson..Network.SiameseModel` node.

* Mom: place a picture of the person which will be _mom_ during the execution in the `mom_img` directory. Write its path (prepending the directory name) in your YML file (`FollowPerson.Mom.ImagePath` node).


**1. Deploy a ROS master**

`roscore`

**2. Connect the computer to the camera stream**

_Turtlebot2_:

`roslaunch openni2_launch openni2.launch`

_Sony EVI D100P_ (modify previously the `resources/usb_cam-test.launch` as indicated above):

`roslaunch usb_cam resources/usb_cam-test.launch`

**3. Launch the actuators drivers**

_Turtlebot2_:

`roslaunch kobuki_node minimal.launch`

_Sony EVI D100P_ (provide r/w permissions to `/dev/ttyUSB0`, as mentioned above):

`evicam_driver evicam_driver.cfg`


**4. Launch the application**

_Turtlebot2_:

`python2 followperson.py turtlebot.yml`

_Sony EVI D100P_:

`python2 followperson.py ptz.yml`

(give it a time to build and load the network instance from the files.)


***
### Object Detector

__Example video:__

[
![YouTube video](http://img.youtube.com/vi/wmtAs7n-r2A/0.jpg)](https://www.youtube.com/watch?v=wmtAs7n-r2A)



This tool was ported to its own repository [(available here)](https://github.com/JdeRobot/dl-objectdetector)


### Digit Classifier

__Example video:__

[
![YouTube video](http://img.youtube.com/vi/x-OhWal38Ak/0.jpg)](https://www.youtube.com/watch?v=x-OhWal38Ak)



This tool was ported to its own repository [(available here)](https://github.com/JdeRobot/dl-digitclassifier)

***


Feel free to [contact me](mailto:nachocmstrato@gmail.com) for further information.
