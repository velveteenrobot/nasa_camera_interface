#include "GlobalDefsDeps.h"
#include "HelperFunctions.h"
#include "SampleDetectorCfg.h"
#include "SampleDetectionState.h"
#include "FrameDefinitions.h"

using namespace cv;

SampleDetectorCfg::SampleDetectorCfg() : FIT_ELLIPSE_MIN_POINTS(5) {

	LoadDefaults();
}

void SampleDetectorCfg::SetPara4LowResImg( Size ImageSize ) {

  int cols = ImageSize.width, rows = ImageSize.height;

  m_ImageWidth  = cols;
  m_ImageHeight = rows;

  m_SMAreaThresh = m_SMAreaThresh * cols * rows;              //!< TUNING_KNOB 

  m_SMAreaThreshAbsMin = m_SMAreaThreshAbsMin * cols * rows;  //!< TUNING_KNOB

  m_SMMaxAreaThresh = m_SMMaxAreaThresh * cols * rows;        //!< TUNING_KNOB 

  m_SMminMinAreaThres = m_SMminMinAreaThres * cols * rows;    //!< TUNING_KNOB 
    
  m_SMmaxMinAreaThres = m_SMmaxMinAreaThres * cols * rows;    //!< TUNING_KNOB 
    
  m_SMminMaxAreaThres = m_SMminMaxAreaThres * cols * rows;    //!< TUNING_KNOB 
    
  m_SMmaxMaxAreaThres = m_SMmaxMaxAreaThres * cols * rows;    //!< TUNING_KNOB         

  m_MaxObjX = m_MaxObjX * cols;
  m_MaxObjY = m_MaxObjY * rows;
}



void SampleDetectorCfg::LoadDefaults() {
	
	SrcObject desiredModels[] = {eObjHook, eObjPuck, eObjBall, eObjTube, eObjHardCylinder, eObjWoodCube, eObjRock };   // TUNING_KNOB
	m_modelVec = set<SrcObject>(desiredModels, desiredModels + sizeof(desiredModels) / sizeof(SrcObject));

	m_showGUI = true;
  m_ImageWidth = 0;
  m_ImageHeight = 0;
  
  minParameterBlobCompactness = 0.9;
  maxParameterBlobCompactness = 12.5;
  // Number coeff for shape descriptor    
  m_FSDesFrequency = 200;
  m_FSDesNumCoeffUsed = 10;
  
  m_maxDistForTempSmoothing = 150.0;

  m_SMMaxAreaThresh = (double) 1000/128640;
  m_SMAreaThresh = (double) 5/128640;
  m_SMAreaThreshAbsMin = ((double)  2/128640);
  m_SMminMinAreaThres = 10.00e-05;
  m_SMmaxMinAreaThres = 50.00e-05;
  m_SMminMaxAreaThres = 200.00e-05;
  m_SMmaxMaxAreaThres = 600.00e-05;
  
  m_MaxObjX = 0.12;	                          //! TUNING_KNOB, blobs can be at most this %ge of horizontal resolution
  m_MaxObjY = 0.3;	                          //! TUNING_KNOB, blobs can be at most this %ge of vertical resolution
  m_NUM_COLOUR_HIST_BINS = 20;                //! TUNING_KNOB, # of CrCb histogram bins for feature descriptor
  m_SIDE_PCT_TO_DUMP = 0.03;	                //! TUNING_KNOB, how much of the horizontal edges to ignore when blob finding
  m_VERT_PCT_TO_DUMP = 0.06;                  //! TUNING_KNOB, how much of the vertical edges to ignore when blob finding
  m_MIN_BLOBS_FOR_HIGHER_MINAREA_THRESH = 2;  //! TUNING_KNOB if there are more than this # of blobs, apply our higher minimum area threshold

  m_vOffset = 480;			                      //!< TUNING_KNOB, how many rows from the top to throw away
  m_botOffset		= 0;
  m_gamma = 2.0;	                            //!< TUNING_KNOB, what gamma value to apply

  m_SMRangeLowThreshold = 0.30;		            //! TUNING_KNOB:, critical parameter !!  Ignore saliency map values lower than this (range 0-1)
  m_SMRefineThreshold = 0.30;                 //! TUNING_KNOB:, critical parameter !!  Ignore saliency map values lower than this (range 0-1)

  m_SM_HISTOGRAM_AREA_THRESH = 0.99;          //!< TUNING_KNOB

  m_ROI_PADDING = 35;		                      //!< TUNING_KNOB, how much of an extra boundary outside the saliency map to search for Y/Cb/Cr edges

  m_DROP_DETECT_AFTER_NFRAMES = 10;           //!< TUNING_KNOB: Reset state if nothing detected in this many frames
  m_CONFIRMED_THRESHOLD = 7;		              //!< TUNING_KNOB: Activate hysteresis after this many detections
  m_MAX_CLOSE_NEIGHBOURS = 2;                 //!<  TUNING_KNOB: Do non-max suppression if we have more than this many neighbours within 
  m_HISTORY_LEN = 30;		                      //!< TUNING_KNOB: Temporal smoothing window, in frames

  m_MAX_FLANN_RESULTS = 50;		                //!< TUNING_KNOB: how many nearest-neighbours to return, max
  m_NMS_RADIUS = 120*120;		                  //!< TUNING_KNOB: Non-max suppression radius, look for neighbours within this many pixels. // Seems to be the squared euclidean distance is needed.

  m_unsharpAlpha = 0.05;                      //!< TUNING_KNOB: default 0.2 -- currently unused !! TODO.
  m_CANNY_UPPER_THRESH = 0.9 * 255;		        //!< TUNING_KNOB:  edge finding, threshold to start identify an edge
  m_CANNY_LOWER_THRESH = m_CANNY_UPPER_THRESH * 0.4;	//!< TUNING_KNOB: threshold to stop following an identified edge

  m_PIXEL_NEIGHBOURHOOD_DIAMETER = FIT_ELLIPSE_MIN_POINTS;  //!< TUNING_KNOB: Recommended value for realtime, bilateral filtering

  m_MAX_BLOB_AREA_EXPANSION = 5;              //!< \var MAX_BLOB_AREA_EXPANSION TUNING_KNOB:  reject blobs whose area has expanded by more than this much during filtering
  m_disableYCbCrMask = 1;
  m_updateGUIeveryNframes = -1;
  m_minSVMconf = 0.62;			// TUNING_KNOB.  1st stage SVM conf threshold.  2nd stage is our temporal smoothing; 3rd stage is the FSM-level
}


bool SampleDetectorCfg::LoadFromFile(std::string filename)
{
	CV_Assert(DoesFileExist(filename));
	
	FileStorage fs(filename, FileStorage::READ);
	fs["m_showGUI"] >> m_showGUI;
	fs["minParameterBlobCompactness"] >>	minParameterBlobCompactness;
	fs["maxParameterBlobCompactness"] >>	maxParameterBlobCompactness;
	fs["m_FSDesFrequency"] >> m_FSDesFrequency;
  fs["m_FSDesNumCoeffUsed"] >> m_FSDesNumCoeffUsed;
  fs["m_maxDistForTempSmoothing"] >> m_maxDistForTempSmoothing;
  fs["m_maxNumOfContMisses"] >> m_maxNumOfContMisses;
  fs["m_minNumOfContTracks"] >> m_minNumOfContTracks;
	fs["m_SMMaxAreaThresh"] >>	m_SMMaxAreaThresh;	
	fs["m_SMAreaThresh"] >>	m_SMAreaThresh;		 
	fs["m_SMAreaThreshAbsMin"] >>	m_SMAreaThreshAbsMin;
	fs["m_SMminMinAreaThres"] >> m_SMminMinAreaThres;
	fs["m_SMmaxMinAreaThres"] >> m_SMmaxMinAreaThres;
	fs["m_SMminMaxAreaThres"] >> m_SMminMaxAreaThres;
	fs["m_SMmaxMaxAreaThres"] >> m_SMmaxMaxAreaThres; 	
	fs["m_MaxObjX"] >>	m_MaxObjX;		 
	fs["m_MaxObjY"] >>	m_MaxObjY;		 
	fs["m_NUM_COLOUR_HIST_BINS"] >> m_NUM_COLOUR_HIST_BINS;		 
	fs["m_SIDE_PCT_TO_DUMP"] >> m_SIDE_PCT_TO_DUMP;		 
	fs["m_VERT_PCT_TO_DUMP"] >> m_VERT_PCT_TO_DUMP;		 
	fs["m_MIN_BLOBS_FOR_HIGHER_MINAREA_THRESH"] >> m_MIN_BLOBS_FOR_HIGHER_MINAREA_THRESH;		 
	fs["m_vOffset"] >> m_vOffset;		 
	fs["m_botOffset"]	>> m_botOffset;
	fs["m_gamma"] >> m_gamma;		 
	fs["m_SMRangeLowThreshold"] >> m_SMRangeLowThreshold;
  fs["m_SMRefineThreshold"]>> m_SMRefineThreshold;	
	fs["m_SM_HISTOGRAM_AREA_THRESH"] >> m_SM_HISTOGRAM_AREA_THRESH;	
	fs["m_ROI_PADDING"] >> m_ROI_PADDING;	
	fs["m_DROP_DETECT_AFTER_NFRAMES"] >> m_DROP_DETECT_AFTER_NFRAMES;	
	fs["m_CONFIRMED_THRESHOLD"] >> m_CONFIRMED_THRESHOLD;	
	fs["m_MAX_CLOSE_NEIGHBOURS"] >> m_MAX_CLOSE_NEIGHBOURS;	
	fs["m_HISTORY_LEN"] >> m_HISTORY_LEN;
	fs["m_MAX_FLANN_RESULTS"] >> m_MAX_FLANN_RESULTS;
	fs["m_NMS_RADIUS"] >> m_NMS_RADIUS;
	fs["m_unsharpAlpha"] >> m_unsharpAlpha;
	fs["m_CANNY_UPPER_THRESH"] >> m_CANNY_UPPER_THRESH;
	fs["m_CANNY_LOWER_THRESH"] >> m_CANNY_LOWER_THRESH;
	fs["m_PIXEL_NEIGHBOURHOOD_DIAMETER"] >> m_PIXEL_NEIGHBOURHOOD_DIAMETER;
	fs["m_MAX_BLOB_AREA_EXPANSION"] >> m_MAX_BLOB_AREA_EXPANSION;
	fs["m_disableYCbCrMask"] >> m_disableYCbCrMask;
	fs["m_updateGUIeveryNframes"] >> m_updateGUIeveryNframes;
	fs["m_minSVMconf"] >> m_minSVMconf;
	fs.release();

	std::cout << "Loaded config from " << filename << std::endl;
	return true;
}


bool SampleDetectorCfg::SaveToFile(std::string filename) 
{
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "m_showGUI" << m_showGUI;
	fs << "minParameterBlobCompactness" <<	minParameterBlobCompactness;
	fs << "maxParameterBlobCompactness" <<	maxParameterBlobCompactness;
	fs << "m_FSDesFrequency" << m_FSDesFrequency;
  fs << "m_FSDesNumCoeffUsed" << m_FSDesNumCoeffUsed;
  fs << "m_maxDistForTempSmoothing" << m_maxDistForTempSmoothing;
  fs << "m_maxNumOfContMisses" << m_maxNumOfContMisses;
  fs << "m_minNumOfContTracks" << m_minNumOfContTracks;
  fs << "m_SMMaxAreaThresh" << m_SMMaxAreaThresh;		
	fs << "m_SMAreaThresh" << m_SMAreaThresh;		 
	fs << "m_SMAreaThreshAbsMin" << m_SMAreaThreshAbsMin;
	fs << "m_SMminMinAreaThres" << m_SMminMinAreaThres;
	fs << "m_SMmaxMinAreaThres" << m_SMmaxMinAreaThres;
	fs << "m_SMminMaxAreaThres" << m_SMminMaxAreaThres;
	fs << "m_SMmaxMaxAreaThres" << m_SMmaxMaxAreaThres;	
	fs << "m_MaxObjX" << m_MaxObjX;		 
	fs << "m_MaxObjY" << m_MaxObjY;		 
	fs << "m_NUM_COLOUR_HIST_BINS" << m_NUM_COLOUR_HIST_BINS;		 
	fs << "m_SIDE_PCT_TO_DUMP" << m_SIDE_PCT_TO_DUMP;		 
	fs << "m_VERT_PCT_TO_DUMP" << m_VERT_PCT_TO_DUMP;		 
	fs << "m_MIN_BLOBS_FOR_HIGHER_MINAREA_THRESH" << m_MIN_BLOBS_FOR_HIGHER_MINAREA_THRESH;		 
	fs << "m_vOffset" << m_vOffset;		
	fs << "m_botOffset"		<< m_botOffset;
	fs << "m_gamma" << m_gamma;		 
	fs << "m_SMRangeLowThreshold" << m_SMRangeLowThreshold;	
  fs << "m_SMRefineThreshold" << m_SMRefineThreshold;
	fs << "m_SM_HISTOGRAM_AREA_THRESH" << m_SM_HISTOGRAM_AREA_THRESH;	
	fs << "m_ROI_PADDING" << m_ROI_PADDING;	
	fs << "m_DROP_DETECT_AFTER_NFRAMES" << m_DROP_DETECT_AFTER_NFRAMES;	
	fs << "m_CONFIRMED_THRESHOLD" << m_CONFIRMED_THRESHOLD;	
	fs << "m_MAX_CLOSE_NEIGHBOURS" << m_MAX_CLOSE_NEIGHBOURS;	

	fs << "m_HISTORY_LEN" << m_HISTORY_LEN;
	fs << "m_MAX_FLANN_RESULTS" << m_MAX_FLANN_RESULTS;
	fs << "m_NMS_RADIUS" << m_NMS_RADIUS;
	fs << "m_unsharpAlpha" << m_unsharpAlpha;
	fs << "m_CANNY_UPPER_THRESH" << m_CANNY_UPPER_THRESH;
	fs << "m_CANNY_LOWER_THRESH" << m_CANNY_LOWER_THRESH;
	fs << "m_PIXEL_NEIGHBOURHOOD_DIAMETER" << m_PIXEL_NEIGHBOURHOOD_DIAMETER;
	fs << "m_MAX_BLOB_AREA_EXPANSION" << m_MAX_BLOB_AREA_EXPANSION;
	fs << "m_disableYCbCrMask" << m_disableYCbCrMask;
	fs << "m_updateGUIeveryNframes" << m_updateGUIeveryNframes;
	fs << "m_minSVMconf" << m_minSVMconf;
	fs.release();
	std::cout << "Wrote config to " << filename << std::endl;

	return true;
}


