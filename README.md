
# Final Degree Project (Nacho Condés)
MediaWiki of this project available [here.](http://jderobot.org/Naxvm-tfg)

# Requirements
* You will need to install JdeRobot for this component to work ([installation guide here](https://jderobot.org/Installation)).
* Make sure you already installed the `ros-kinetic-usb-cam`. You can easily install it via <code>apt</code> once you have already installed JdeRobot:
`sudo apt install ros-kinetic-usb-cam`
* All the necessary Python packages have been annotated for <code>pip</code> to install them automatically. To do so, run:
`pip2 install -r requirements.txt`

## Follow Person (current stuff):
A component which commands a Sony Evicam D100P to track a person with the use of Deep Learning (a Convolutional Neural Network implemented on TensorFlow or Keras (functional, but not usable, due to its slowness: it's 8 times slower than TensorFlow)).

__Video available soon!__

### Available detection models:
You can download any of [these pre-trained models](http://jderobot.org/store/deeplearning-networks/TensorFlow/) to embed them on the component. Just download the model (<code>pb</code> file for TF, <code>h5</code> for Keras), place it on the <code>Network/[desired framework]</code> directory, and specify its name on the <code>followperson.yml</code> file (<code>Network.Model</code> node of the YML tree).

### How to execute:
Make sure that you execute `sudo chmod 777 /dev/ttyUSB0` when you connect the PT motors (EVI connector) to your computer (`evicam_driver` needs this to be this way, otherwise it will raise an _EBADF_ error when trying to access the device).

Also, check which of your computer video devices corresponds to the PT camera interface. You can perform this launching `ls /dev`. You will see the devices related to your computer. `/dev/video0` is tipically your laptop webcam (or default camera). The PT camera will correspond to the next device, which can stand for `/dev/video1`, `/dev/video2`, etc. This is due to the order of the USB connections. You will have to change the value of the `usb_cam-test.launch` file to match to this device no (line 2):
  `<param name="video_device" value="/dev/your_video_device" />`

Once this is done, you are ready to rock!
* Terminal 1:
`$ roslaunch usb_cam-test.launch`
* Terminal 2:
`$ evicam_driver evicam_driver.cfg`
* Terminal 3:
`$ python2 followperson.py followperson.yml`





## Digit Classifier:
Node (ported to its own [official JdeRobot repository](https://github.com/JdeRobot/dl-digitclassifier)) capable of classifying a real-time image into 10 categories, corresponding to digits from 0 to 9. Video below.

__Available on [this release](https://github.com/RoboticsURJC-students/2017-tfg-nacho_condes/releases/tag/digit_classifier).__
### How to execute:
* Terminal 1:
`$ cameraserver cameraserver.cfg`
*  Terminal 2:
`$ python2 digitclassifier.py digitclassifier.yml`


Video example:

[
![YouTube video](http://img.youtube.com/vi/x-OhWal38Ak/0.jpg)](http://www.youtube.com/watch?v=x-OhWal38Ak)
