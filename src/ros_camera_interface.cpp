#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include <nasa_camera_interface/image_info.h>

//This is just to check if file exists.
//If you don't want boost you can just check however
#include <boost/filesystem.hpp>

//declare camera parameters here
std::string camera_param;
int camera_id = 1;
ros::Publisher image_pub1, image_pub2, image_pub3;



void spinOnce(ros::Rate& loopRate) {
  loopRate.sleep(); //Maintain the loop rate
  ros::spinOnce();   //Check for new messages
}

void trigger(const ros::TimerEvent&)
{
  //Use parameters to trigger camera?
  /* Code goes here 
     This code comes from the gphoto stuff? 
     Ask Shahid...
  */

  //Retrieve image info from filename?

  nasa_camera_interface::image_info image_information;
  image_information.path = "path";
  image_information.image_id = 1;

  

  switch (camera_id)
  {
	case 1:
	  image_pub1.publish(image_information);
	  camera_id = 2;
	  break;
	case 2:
	  image_pub2.publish(image_information);
	  camera_id = 3;
	  break;
	case 3:
    image_pub3.publish(image_information);
    camera_id = 1;
    break;
  }

  //image_pub.publish(camera_image);

  //spinOnce(loopRate);
}

bool LoadFromFile(std::string filename)
{
	//This is just to check if file exists.
  //If you don't want boost you can just check however
	if (boost::filesystem::exists(filename))
	{
		cv::FileStorage fs(filename, cv::FileStorage::READ);

	  //Load parameters here
		fs["camera_param"] >> camera_param;
		
		fs.release();

		std::cout << "Loaded config from " << filename << std::endl;
		return true;
	}
	else
		return false;
	
	
}


bool SaveDefaults(std::string filename) 
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);

	//Write default parameters here 
	fs << "camera_param" << camera_param;
	
	fs.release();
	std::cout << "Wrote config to " << filename << std::endl;

	return true;
}


int main(int argc, char **argv)
{  
  //Initialise node
  ros::init(argc,argv,"camera_interface_node");
  ros::NodeHandle n;
  ros::Rate loopRate(20);

  //Create publishers
  image_pub1 = n.advertise<nasa_camera_interface::image_info>(
    "/camera/rgb/image_info/image1",1);
  image_pub2 = n.advertise<nasa_camera_interface::image_info>(
    "/camera/rgb/image_info/image2",1);
  image_pub3 = n.advertise<nasa_camera_interface::image_info>(
    "/camera/rgb/image_info/image3",1);

  //Read in camera parameters from file? - use yaml parsing, if file doesn't exist, make default
  std::string filename = "filename.txt";
  if (!LoadFromFile(filename))
  {
  	SaveDefaults(filename);
  }
  
  /* Could also read from yaml file into dict, etc.
     Just need to specify format
  */

  //Initialise cameras


  //Do this while node is active
  while (ros::ok())
  {
  	ros::Timer timer = n.createTimer(ros::Duration(0.11), trigger);
  	ros::spin();
  }


}