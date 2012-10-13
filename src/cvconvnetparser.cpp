/*****************************************************************************
 IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING. By
downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install, copy or
use the software.

Contributors License Agreement

Copyright© 2007, Akhmed Umyarov. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
- The name of Contributor may not be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
All information provided related to future Intel products and plans is
preliminary and subject to change at any time, without notice.
*****************************************************************************/

/*!\file
 * \brief XML Parser implementation
 *
 * The file contains an XML parser that parses 
 * a file using Expat library and 
 * creates convolutional network.
 * 
 * The implementation is not particularly easy to read,
 * partially because Expat is a plain C library and SaX parser.
 * The reason for choosing Expat is that it is available both for
 * Windows and for Linux, and it is light-weight unlike most of the
 * DOM-parsers.
 * \author Akhmed Umyarov
 * \date 2007
 */

#include <expat.h> // XML Parsing
#include <cassert>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <iterator>
#include <sstream>
#include "cvconvnetparser.h"
#include "cvconvolutionplane.h"
#include "cvgenericplane.h"
#include "cvmaxplane.h"
#include "cvmaxoperatorplane.h"
#include "cvrbfplane.h"
#include "cvregressionplane.h"
#include "cvsourceplane.h"
#include "cvsubsamplingplane.h"


// Bit masks for tag processing
const int INSIDE_TAG=1<<0;
const int FOUND_VALUE=1<<1;

	
using namespace std;

// ***********************************************************************
// ******************** Expat XML Parsing handlers ***********************
// ***********************************************************************

//! The structure used by Expat XML parser
/*!
 * This is the storage used by Expat XML parser for storing
 * current state of SAX XML parsing.
 */
typedef struct
{
	// Text parameters
	string &creator;	//!< Network creator
	string &name;		//!< Network name
	string &info;		//!< Network info

 	// Graph parameters
	vector<CvGenericPlane *> &plane; //!< Container for all planes
	map<string,int> &idmap; //!< Mapping between ids

	// Current (recursive) parameters
	int depth;			//!< Current depth of XML recursion
	int isbias;			//!< bit mask for "bias" tag
	int isinfo;			//!< bit mask for "info" tag
	int isconnection;		//!< bit mask for "connection" tag
	vector<double> cur_weight;	//!< Weights for current plane
	vector<CvGenericPlane *> cur_parents; //!< Parents for current plane
	string cur_type;		//!< Current plane type
	
	// Parser 
	XML_Parser &parser;		//!< Pointer to parser struct
} XMLparserData;

//! Function that stops parser in case of error
static void XMLCALL icvXML_StopParser(XML_Parser &parser, string errstr)
{
	cout << "XML Error: " << errstr << endl;
	XML_StopParser(parser,XML_FALSE); // Stop parser unresumably
}

//! Macro for checking XML conditions
#define CHK_POSSIBLE_FAIL(x,y) if ( x ) { \
	icvXML_StopParser(data.parser,(y));\
	return; }

//! SAX callback function invoked when new open tag is encountered
static void XMLCALL icvXML_StartElementHandler (void *userData, const XML_Char *name, const XML_Char **atts)
{
	assert( userData != NULL && name != NULL );

	XMLparserData &data = *((XMLparserData *) userData);
	string namestr (name);
	
	if ((namestr == "net") && (data.depth == 0))
	// ****** Process <net> tag
	{
		// Remove all existing planes
		for (signed int i = data.plane.size()-1; i >= 0; i--)
		{
			assert ( data.plane[i] != NULL );
			delete data.plane[i];
		}
		
		// Clear all the structures;
		data.plane.clear();
		data.creator.clear();
		data.name.clear();
		data.idmap.clear();
		data.isbias = 0;
		data.isinfo = 0;

		for (int i=0; (atts[i]!=NULL) && (atts[i+1]!=NULL); i+=2)
		{
			string attr = atts[i];
			string val = atts[i+1];

			if (attr=="creator")
				data.creator = val;
			if (attr=="name")
				data.name = val;
		}
	}
	else if ((namestr == "plane") && (data.depth == 1))
	// ****** Process <plane> tag	
	{
		int curplaneid = data.plane.size(); // Planeid that is going to be assigned for this plane
		string planeid,planetype; // Plane info as read from the files
		int fmapszx = 0, fmapszy = 0, neuroszx = 0, neuroszy = 0;

		// Initialize data structures
		data.cur_parents.clear();
		data.cur_weight.clear();	
		data.isbias = 0;
		data.isconnection = 0;

		for (int i=0; (atts[i]!=NULL) && (atts[i+1]!=NULL); i+=2)
		{
			string attr = atts[i];
			string val = atts[i+1];
			if (attr=="id") planeid = val;
			if (attr=="type") planetype = val;
			if (attr=="featuremapsize")  
			{
				istringstream iss ( val );
				char c;
				iss >> fmapszx >> c >> fmapszy;
			}
			if (attr=="neuronsize")  
			{
				istringstream iss ( val );
				char c;
				iss >> neuroszx >> c >> neuroszy;
			}
		}
		// Plane MUST have an id
		CHK_POSSIBLE_FAIL( planeid.size()==0 , "plane has no id");

		// For all planes except MAX check whether featuremap sizes and neuron sizes are consistent
		if (planetype != "max" && planetype != "regression")
		{
			CHK_POSSIBLE_FAIL( (fmapszx <= 0 || fmapszy <= 0 || fmapszx > CVCONVOLUTIONALNET_MAX_FMAPSZ || fmapszy > CVCONVOLUTIONALNET_MAX_FMAPSZ) ,"feature map size is inconsistent");
                }
                if (planetype != "max")
                {
	
			CHK_POSSIBLE_FAIL( (neuroszx < 0 || neuroszy < 0 || neuroszx > CVCONVOLUTIONALNET_MAX_FMAPSZ || neuroszy > CVCONVOLUTIONALNET_MAX_FMAPSZ), "neuron window size is inconsistent");
		}
				
		// Create required plane object with inited parameters 
		if (planetype=="source")
		{
			data.plane.push_back( 
				new CvSourcePlane(planeid,cvSize(fmapszx,fmapszy)) 
			);
			data.idmap[planeid] = curplaneid;
		} else if (planetype=="convolution")
		{
			data.plane.push_back(
				new CvConvolutionPlane(planeid,cvSize(fmapszx,fmapszy),cvSize(neuroszx,neuroszy))
			);
			data.idmap[planeid] = curplaneid;
		} else if (planetype=="subsampling")
		{
			data.plane.push_back(
				new CvSubSamplingPlane(planeid,cvSize(fmapszx,fmapszy),cvSize(neuroszx,neuroszy))
			);
			data.idmap[planeid] = curplaneid;
                } else if (planetype=="maxoperator")
                {
                        data.plane.push_back(
                                new CvMaxOperatorPlane(planeid,cvSize(fmapszx,fmapszy),cvSize(neuroszx,neuroszy))
                        );
			data.idmap[planeid] = curplaneid;
		} else if (planetype=="rbf")
		{
			data.plane.push_back(
				new CvRBFPlane(planeid,cvSize(fmapszx,fmapszy),cvSize(neuroszx,neuroszy))
			);
			data.idmap[planeid] = curplaneid;
		} else if (planetype=="max")
		{
			data.plane.push_back(
				new CvMaxPlane(planeid)
			);
			data.idmap[planeid] = curplaneid;
		} else if (planetype=="regression")
                {
                        data.plane.push_back(
                                new CvRegressionPlane(planeid,cvSize(neuroszx,neuroszy))
                        );
			data.idmap[planeid] = curplaneid;
                } else
		{
			CHK_POSSIBLE_FAIL(1, "plane "+planeid+" has no type or unidentified type");			
		}
		data.cur_type = planetype;
		data.cur_weight.clear();
	} else if ((namestr == "connection") && (data.depth==2))
	// ****** Process <connection> tag	
	{
		int curplaneid = data.plane.size()-1; // Planeid of the current plane

		for (int i=0;  (atts[i]!=NULL) && (atts[i+1]!=NULL); i+=2)
		{
			string attr = atts[i];
			string val = atts[i+1];

			if (attr=="to")
			{
				// Find the plane number by the plane id
				map<string,int>::iterator itr = data.idmap.find(val); 
				
				CHK_POSSIBLE_FAIL (itr == data.idmap.end(), "Connection to non-existing plane or graph is not topologically sorted! Attempt to connect to \""+val+"\"");
				
				data.cur_parents.push_back(data.plane[itr->second]);
			}
		}
		data.isconnection |= INSIDE_TAG;
	}
	else if ((namestr == "bias") && (data.depth==2))
	{
		data.isbias |= INSIDE_TAG; // Mark as inside <bias> tag
		
		CHK_POSSIBLE_FAIL(data.cur_type=="max","<bias> defined for max plane");
	}
	else if ((namestr == "info") && (data.depth == 1))
	{
		data.isinfo |= INSIDE_TAG; // Mark as inside <info> tag

	} else
	{
		// Undefined xml tag found
		CHK_POSSIBLE_FAIL(1,"undefined or misplaced tag <"+namestr+"> found");
	}
	
	data.depth++;
}

//! SAX callback function invoked when a tag is closed
static void XMLCALL icvXML_EndElementHandler(void *userData, const XML_Char *name)
{
	assert( userData != NULL && name != NULL);

	XMLparserData &data = *((XMLparserData *) userData);
	data.depth--;	

	string namestr (name);

	if ((namestr == "plane") && (data.depth == 1))
	{
		// Get current plane id
		vector<CvGenericPlane *>::iterator i = data.plane.end()-1;
		
		// Check if we found <bias> for certain planes
		CHK_POSSIBLE_FAIL( (!(data.isbias & FOUND_VALUE)) && (data.cur_type=="convolution" || data.cur_type=="subsampling"), "no bias found");

		// Check if plane (except source) is connected to something
		CHK_POSSIBLE_FAIL( (data.cur_parents.size()==0) && (data.cur_type!="source"), "plane is not connected to anything");
		
		// Connect to parent planes
		CHK_POSSIBLE_FAIL( !(*i)->connto(data.cur_parents), "failed to accomplish connections");

		// Assign weights that we have read so far
		CHK_POSSIBLE_FAIL( !(*i)->setweight(data.cur_weight), "failed to assign weights");
		
		// Clear data structures
		data.cur_parents.clear();
		data.cur_weight.clear();	
		data.isbias = 0;
		data.isconnection = 0;
	} else if ((namestr == "connection") && (data.depth == 2))
	{
		data.isconnection &= ~INSIDE_TAG;
	} else if ((namestr == "bias") && (data.depth == 2))
	{
		data.isbias &= ~INSIDE_TAG;
	} else if ((namestr == "info") && (data.depth == 1))
	{
		data.isinfo &= ~INSIDE_TAG;
	}
}

//! SAX callback function invoked when XML data are encountered
static void XMLCALL icvXML_CharacterDataHandler(void *userData, const XML_Char *s, int len)
{
	assert( userData != NULL && s != NULL);

	XMLparserData &data = *((XMLparserData *) userData);

	string weights(s,len);
	istringstream iss(weights);
	
	double w;
	if (data.isbias & INSIDE_TAG) 
	// Process <bias> tag data
	{
		if (iss >> w)
		{
			data.cur_weight.insert(data.cur_weight.begin(),w);
			data.isbias |= FOUND_VALUE; // Mark as found
		}
		
	} else if (data.isinfo & INSIDE_TAG)
	// Process <info> tag data
	{
		iss >> data.info;
		data.isinfo |= FOUND_VALUE; // Mark as found
	} else if (data.isconnection & INSIDE_TAG)
	// Process <connection> tag data
	{
		while (iss >> w)
		{
			CHK_POSSIBLE_FAIL(data.cur_type=="max","weights defined for MAX plane. Nonsense!");
			data.cur_weight.push_back(w);
		}
		data.isconnection |= FOUND_VALUE;  // Mark as found
	};
	// Ignore other data in XML
}

//! Parser initialization and parsing invocation
int parse(string xml, string &creator,
		string &name, 
		string &info, 
 		vector<CvGenericPlane *> &plane,
		map<string,int> &idmap)
{
	XML_Parser parser = XML_ParserCreate(NULL);

	// Data that are to be passed to handler
	XMLparserData handlerdata = 
	{
		creator, // string &creator;
		name, // string &name;
		info, // string &info;

 		plane, // vector<CvGenericPlane *> &plane;
		idmap, // map<string,int> &idmap;

		0, // int depth;
		0, // int isbias;
		0, // int isinfo;
		0, // int isconnection;
		
		vector<double> (), //vector<double> cur_weight;	
		vector<CvGenericPlane *> (), // vector<CvGenericPlane *> cur_parents; 
		"", // string cur_type;
		parser
	}; 

	// SAX initialization
	XML_SetUserData(parser,&handlerdata);
	XML_SetElementHandler(parser, icvXML_StartElementHandler,  icvXML_EndElementHandler);
	XML_SetCharacterDataHandler(parser, icvXML_CharacterDataHandler);

	int errcode = 1;
	
	// Main parsing
	if ( XML_Parse(parser, xml.c_str(), xml.size(), 1) == XML_STATUS_ERROR )
	{
		cerr << "Error parsing the XML: " << XML_ErrorString( XML_GetErrorCode(parser) ) << " at line " << XML_GetCurrentLineNumber(parser) << endl;
		errcode = 0;
	}
	XML_ParserFree(parser);

	return errcode;
}
