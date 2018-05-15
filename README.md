
# Final Degree Project (Nacho Condés)
MediaWiki of this project available [here.](http://jderobot.org/Naxvm-tfg)

## Follow Person (current stuff):
A component which commands a Sony Evicam 100D to track a person with the use of Deep Learning (a Convolutional Neural Network implemented on TensorFlow or ~~Keras~~ (not yet)).

### How to execute:
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
